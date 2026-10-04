import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.ksdd2_robustness.metrics import PixelSpools
from inspection.ksdd2 import build_manifest
from inspection.ksdd2_evaluate import evaluate as stock_evaluate
from inspection.model import SegmentationOnlyModel
from inspection.preprocessing import prepare_image
from scripts import ksdd2_native_stream as stream
from test_ksdd2 import fixture_data


@pytest.fixture
def fixture_checkpoint(tmp_path):
    fixture_data(tmp_path / "dataset")
    # Different original aspect ratios exercise independent transforms per batch.
    for suffix in ("", "_GT"):
        path = tmp_path / f"dataset/test/20001{suffix}.png"
        with Image.open(path) as source:
            resized = source.resize((30, 40), Image.Resampling.NEAREST)
        resized.save(path)
    manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.manual_seed(42)
    checkpoint = {"schema_version": 2, "model_kind": "supervised_segmentation",
                  "category": "kolektor_surface", "model_config": {"base_channels": 2},
                  "model_state": SegmentationOnlyModel(2).state_dict(), "threshold": .5,
                  "preprocess": {"mode": "letterbox", "height": 48, "width": 32},
                  "split_digest": manifest["split_digest"], "config": {"training_seed": 42},
                  "calibration": {"normal_count": 1}, "threshold_source": "normal calibration",
                  "model_selection": "pooled validation harmonic mean", "best_epoch": 5, "best_metric": .8}
    torch.save(checkpoint, checkpoint_path)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield manifest_path, checkpoint_path, manifest, checkpoint
    finally:
        torch.set_num_threads(previous_threads)


def test_streaming_matches_stock_cpu_native_geometry_and_frozen_decisions(fixture_checkpoint, tmp_path, monkeypatch):
    manifest_path, checkpoint_path, manifest, checkpoint = fixture_checkpoint
    other_splits = {record[key] for split, records in manifest["splits"].items() if split != "test"
                    for record in records for key in ("image", "mask")}
    original_open = Image.open

    def only_test_pixels(path, *args, **kwargs):
        assert str(path) not in other_splits, "Evaluation opened optimization/calibration pixels"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", only_test_pixels)
    stock = stock_evaluate(manifest_path, checkpoint_path, device="cpu", batch_size=2,
                           bootstrap_samples=10, experimental_status="exploratory")
    scratch = tmp_path / "scratch"
    result = stream.evaluate(manifest_path, checkpoint_path, device="cpu", batch_size=2,
                             bootstrap_samples=10, scratch_root=scratch)
    assert result["predictions"] == stock["predictions"]
    for name in ("image_auroc", "image_average_precision", "pixel_average_precision"):
        assert result[name] == pytest.approx(stock[name], abs=1e-12)
    for name in ("confusion", "defect_recall", "normal_false_alarm_rate", "precision",
                 "defect_area_groups", "recall_by_defect", "threshold", "class_prevalence",
                 "split_digest", "pixel_metric_space", "pixel_average_precision_by_category"):
        assert result[name] == stock[name]
    assert result["experimental_status"] == "exploratory"
    assert result["evaluation_resolution"] == "original resolution per image"
    assert result["validation_selection"] == checkpoint["model_selection"]
    assert result["uncertainty"]["image_auroc_interval"] == stock["uncertainty"]["image_auroc_interval"]
    assert result["uncertainty"]["defect_recall_interval"] == stock["uncertainty"]["defect_recall_interval"]
    assert "prior test inspection" in result["uncertainty"]["limitations"]
    assert "latency" not in result and "gallery" not in result
    assert not list(scratch.iterdir())
    root = Path(__file__).resolve().parents[1]
    for name in ("scripts/ksdd2_native_stream.py", "experiments/ksdd2_robustness/metrics.py"):
        assert result["evaluation_additional_source_sha256"][name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert "engine.py" in result["evaluation_source_sha256"]
    json.dumps(result, allow_nan=False)


def test_native_spool_crops_padding_and_preserves_unchanged_mask(tmp_path):
    image = Image.new("RGB", (8, 32), "gray")
    target = np.zeros((32, 8), dtype=np.uint8)
    target[10:14, 2:6] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(target).save(mask_path)
    _, _, transform = prepare_image(image, {"mode": "letterbox", "height": 32, "width": 32})
    activation = np.ones((32, 32), dtype=np.float32)  # Large values outside the true image.
    activation[:, transform["left"]:transform["left"] + 8] = target / 255
    spool = PixelSpools(tmp_path / "spool", conditions=1)
    try:
        stream._append_native(spool, {"mask": str(mask_path), "original_size": [8, 32]}, activation, transform)
        pooled, conditions, counts = spool.metrics()
        assert pooled == conditions[0] == 1.
        assert counts == [{"negative": 240, "positive": 16}]
    finally:
        spool.close()


def test_native_spool_rejects_misaligned_mask(tmp_path):
    image = Image.new("RGB", (8, 32), "gray")
    mask_path = tmp_path / "mask.png"
    Image.new("L", (9, 32), 0).save(mask_path)
    _, _, transform = prepare_image(image, {"mode": "letterbox", "height": 32, "width": 32})
    spool = PixelSpools(tmp_path / "spool", conditions=1)
    try:
        with pytest.raises(ValueError, match="must align"):
            stream._append_native(spool, {"mask": str(mask_path), "original_size": [8, 32]},
                                  np.zeros((32, 32), dtype=np.float32), transform)
    finally:
        spool.close()


def test_failure_cleans_spools(fixture_checkpoint, tmp_path, monkeypatch):
    manifest_path, checkpoint_path, _, _ = fixture_checkpoint

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interrupted map append")

    monkeypatch.setattr(stream, "_append_native", interrupted)
    scratch = tmp_path / "scratch"
    with pytest.raises(RuntimeError, match="simulated"):
        stream.evaluate(manifest_path, checkpoint_path, device="cpu", bootstrap_samples=2, scratch_root=scratch)
    assert not list(scratch.iterdir())


def test_split_mismatch_rejected_before_test_content(fixture_checkpoint, monkeypatch):
    manifest_path, checkpoint_path, _, checkpoint = fixture_checkpoint
    checkpoint["split_digest"] = "incompatible"
    torch.save(checkpoint, checkpoint_path)
    original_load = stream.load_manifest

    def metadata_only(path, verify_splits=()):
        assert not verify_splits, "Test content accessed before compatibility check"
        return original_load(path, verify_splits)

    monkeypatch.setattr(stream, "load_manifest", metadata_only)
    with pytest.raises(ValueError, match="split differ"):
        stream.evaluate(manifest_path, checkpoint_path, device="cpu")
