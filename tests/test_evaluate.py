import json

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.data import build_manifest
from inspection.engine import InspectionEngine
from inspection.evaluate import benchmark_checkpoint, compute_metrics, evaluate_checkpoint, original_pixel_metrics, stress_test_checkpoint
from inspection.model import InspectionModel


def test_frozen_threshold_and_ranking_are_different():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.7, 0.9]
    masks = np.zeros((4, 1, 2, 2), dtype=np.uint8)
    masks[2:, :, 0, 0] = 1
    maps = np.zeros_like(masks, dtype=np.float32)
    maps[masks.astype(bool)] = 0.9
    result = compute_metrics(labels, scores, masks, maps, threshold=0.8)
    assert result["image_auroc"] == 1.0
    assert result["pixel_average_precision"] == 1.0
    assert result["confusion"] == {
        "true_positive": 1, "false_positive": 0, "true_negative": 2, "false_negative": 1}
    assert result["defect_recall"] == 0.5
    assert result["threshold"] == 0.8


def test_absent_classes_produce_explicit_null_metrics():
    result = compute_metrics([0, 0], [0.1, 0.8], np.zeros((2, 1, 2, 2)),
                             np.zeros((2, 1, 2, 2)), threshold=0.5)
    assert result["image_auroc"] is None
    assert result["pixel_average_precision"] is None
    assert result["defect_recall"] is None
    assert result["normal_false_alarm_rate"] == 0.5


def test_nonfinite_predictions_and_misaligned_masks_rejected():
    with pytest.raises(ValueError, match="finite"):
        compute_metrics([0, 1], [0.1, np.nan], np.zeros((2, 1, 2, 2)),
                        np.zeros((2, 1, 2, 2)), threshold=0.5)
    with pytest.raises(ValueError, match="aligned"):
        compute_metrics([0, 1], [0.1, 0.9], np.zeros((2, 1, 2, 2)),
                        np.zeros((2, 1, 4, 4)), threshold=0.5)


def test_original_pixel_space_preserves_tiny_defect(tmp_path):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[0, 0] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask).save(path)
    assert not np.asarray(Image.fromarray(mask).resize((128, 128), Image.Resampling.NEAREST)).any()
    maps = np.zeros((1, 1, 128, 128), dtype=np.float32)
    maps[0, 0, 0, 0] = 1
    result = original_pixel_metrics([{"category": "metal_nut", "mask": str(path),
                                       "original_size": [512, 512]}], maps)
    assert result["pixel_metric_space"] == "original"
    assert result["pixel_average_precision_by_category"]["metal_nut"]["positive_pixels"] == 1
    assert result["pixel_average_precision"] > 0


def test_model_space_pixel_metrics_exclude_letterbox_padding():
    masks = np.asarray([[[[0, 1, 0, 0]]]], dtype=np.float32)
    maps = np.asarray([[[[0.1, 0.9, 1.0, 1.0]]]], dtype=np.float32)
    valid = np.asarray([[[[True, True, False, False]]]])
    result = compute_metrics([1], [0.9], masks, maps, threshold=0.5, valid_masks=valid)
    assert result["pixel_average_precision"] == 1.0


def _evaluation_fixture(tmp_path):
    root = tmp_path / "source"
    random = np.random.default_rng(42)
    for index in range(12):
        path = root / "metal_nut/train/good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(random.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(path)
    for defect in ("good", "scratch"):
        path = root / "metal_nut/test" / defect / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(random.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(path)
    path = root / "metal_nut/ground_truth/scratch/000_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[3:6, 3:6] = 255
    Image.fromarray(mask).save(path)
    manifest, _ = build_manifest(root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    model = InspectionModel(base_channels=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"schema_version": 2, "model_state": model.state_dict(), "model_config": {"base_channels": 4},
                "config": {"category": "metal_nut"}, "image_size": 64, "threshold": 0.6,
                "preprocess": {"mode": "letterbox", "height": 64, "width": 64},
                "split_digest": manifest["split_digest"]}, checkpoint_path)
    return manifest_path, checkpoint_path, manifest


def test_evaluation_matches_inspection_engine_and_restores_letterbox(tmp_path):
    manifest_path, checkpoint_path, manifest = _evaluation_fixture(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = evaluate_checkpoint(manifest_path, checkpoint_path, device_name="cpu",
                                     overlay_directory=tmp_path / "gallery", batch_size=2, bootstrap_samples=20)
        engine = InspectionEngine(checkpoint_path, "cpu")
        for record, prediction in zip(manifest["splits"]["test"], result["predictions"]):
            inspected = engine.inspect(record["image"])
            assert inspected["summary"]["score"] == prediction["score"]
            assert inspected["summary"]["predicted_defective"] == prediction["predicted_defective"]
            assert inspected["native_map"].shape == (32, 48)
        assert result["gallery"]["groups"] == {"FP": 0, "FN": 1, "TP": 0, "TN": 1}
        assert result["checkpoint_sha256"] == engine.checkpoint_sha256
        assert result["model_sha256"] == engine.model_sha256
        assert result["pixel_average_precision_by_category"]["metal_nut"]["pixels"] == 2 * 32 * 48
        latency = benchmark_checkpoint(checkpoint_path, manifest["splits"]["calibration"][0]["image"],
                                       "cpu", measurements=3, warmup=1)
        assert latency["measurements"] == 3
        assert latency["device"] == "cpu"
        assert latency["checkpoint_sha256"] == engine.checkpoint_sha256
        assert latency["single_image_p95_ms"] >= latency["single_image_median_ms"] > 0
    finally:
        torch.set_num_threads(previous_threads)


def test_stress_evaluation_keeps_checkpoint_threshold_and_pairs_images(tmp_path):
    manifest_path, checkpoint_path, _ = _evaluation_fixture(tmp_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        stress = stress_test_checkpoint(manifest_path, checkpoint_path, "cpu", batch_size=2)
        assert len(stress["results"]) == 5
        assert all(result["threshold"] == 0.6 for result in stress["results"].values())
        assert all(result["decision_flips"] == 0 for result in stress["results"].values())
        assert stress["results"]["original"]["images"] == 2
    finally:
        torch.set_num_threads(previous_threads)
