"""Synthetic CPU checkpoint/fixture tests only; no production weights or images."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from sklearn.metrics import average_precision_score
import torch

from experiments.ksdd2_decision.model import make_model, parameter_counts
from experiments.ksdd2_decision.protocol import config, source_hashes, SELECTION_RULE
from experiments.ksdd2_robustness.metrics import PixelSpools
from inspection.checkpointing import config_digest
from inspection.engine import InspectionEngine
from inspection.ksdd2 import build_manifest
from inspection.preprocessing import prepare_image, restore_map
from scripts import ksdd2_decision_inference as inference
from scripts import ksdd2_decision_native as native
from test_ksdd2 import fixture_data


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoints(tmp_path):
    fixture_data(tmp_path / "dataset")
    for suffix in ("", "_GT"):
        path = tmp_path / f"dataset/test/20001{suffix}.png"
        with Image.open(path) as image:
            resized = image.resize((30, 40), Image.Resampling.NEAREST)
        resized.save(path)
    manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    paths, states = {}, {}
    for candidate in ("control", "decision_head"):
        cfg = {**config(candidate, "cpu"), "base_channels": 2, "epochs": 2,
               "selection_every": 1, "patience_epochs": 2,
               "preprocess": {"mode": "letterbox", "height": 48, "width": 32}}
        torch.manual_seed(42)
        model = make_model(cfg).eval()
        if candidate == "decision_head":
            with torch.no_grad():
                model.head.output.bias.sub_(10.)
        scores = []
        for record in manifest["splits"]["calibration"]:
            if record["label"]:
                continue
            with Image.open(record["image"]) as image:
                tensor, valid, _ = prepare_image(image.convert("RGB"), cfg["preprocess"])
            with torch.no_grad():
                _, value = model(tensor[None], valid[None])
            scores.append(float(value[0]))
        threshold = float(np.quantile(scores, .9, method="linear"))
        checkpoint = {"schema_version": 2, "checkpoint_kind": "inference", "resumable": False,
            "model_kind": model.model_kind, "model_config": {"base_channels": 2},
            "decision_model_config": {"candidate": candidate, "base_channels": 2, "head_seed": cfg["head_seed"]},
            "parameter_counts": parameter_counts(model), "model_state": model.state_dict(),
            "category": "kolektor_surface", "preprocess": cfg["preprocess"], "config": cfg,
            "config_digest": config_digest(cfg), "manifest_digest": manifest["digest"],
            "split_digest": manifest["split_digest"], "bank_digest": "b" * 64,
            "declaration_digest": "d" * 64, "score_definition": model.score_definition,
            "model_selection": SELECTION_RULE,
            "training_kind": "random_initialization_real_defect_supervision_detached_decision_screen",
            "test_status": "development-inspected/exploratory for all reused KSDD2 tests",
            "best_epoch": 1, "completed_epochs": 2, "best_metric": .8,
            "best_validation": {"pooled": {"image_pauc": .8}, "bank_digest": "b" * 64,
                "split_digest": manifest["split_digest"], "model_kind": model.model_kind,
                "score_definition": model.score_definition, "selection_rule": SELECTION_RULE},
            "provenance": {"decision_source_files": source_hashes(), "config_digest": config_digest(cfg),
                "split_digest": manifest["split_digest"], "bank_digest": "b" * 64},
            "threshold": threshold,
            "calibration": {"scores": scores, "normal_count": len(scores), "split": "calibration", "quantile": .9,
                "quantile_method": "linear", "decision_rule": ">=", "threshold": threshold,
                "calibration_exceedances": sum(value >= threshold for value in scores),
                "model_kind": model.model_kind, "score_definition": model.score_definition},
            "calibration_identity": {"selected_checkpoint_sha256": "e" * 64, "gate_digest": "f" * 64},
            "calibration_status": "fit once after immutable two-run validation gate passed",
            "threshold_source": "separate original clean normal calibration split"}
        path = tmp_path / f"{candidate}.pt"
        torch.save(checkpoint, path)
        paths[candidate], states[candidate] = path, checkpoint
    return manifest_path, manifest, paths, states


def prepared_batch(records, preprocess):
    prepared = []
    for record in records:
        with Image.open(record["image"]) as image:
            prepared.append(prepare_image(image.convert("RGB"), preprocess))
    return torch.stack([p[0] for p in prepared]), torch.stack([p[1] for p in prepared]), [p[2] for p in prepared]


def test_new_control_is_exactly_shared_engine_scores_and_maps(checkpoints):
    _, manifest, paths, _ = checkpoints
    adapted = inference.DecisionInspectionEngine(paths["control"], "cpu")
    shared = InspectionEngine(paths["control"], "cpu")
    images, valid, _ = prepared_batch(manifest["splits"]["test"], adapted.preprocess)
    expected_maps, expected_scores = shared.predict_tensor(images, valid)
    maps, scores = adapted.predict_tensor(images, valid)
    assert torch.equal(maps, expected_maps)
    assert torch.equal(scores, expected_scores)
    assert adapted.display_max == 1.


def test_negative_head_score_tensor_inspect_and_forward_are_identical(checkpoints):
    _, manifest, paths, states = checkpoints
    engine = inference.DecisionInspectionEngine(paths["decision_head"], "cpu")
    assert engine.threshold < 0
    images, valid, _ = prepared_batch(manifest["splits"]["test"], engine.preprocess)
    model = make_model(states["decision_head"]["config"]); model.load_state_dict(states["decision_head"]["model_state"]); model.eval()
    with torch.no_grad():
        expected_logits, expected_scores = model(images, valid)
    maps, scores = engine.predict_tensor(images, valid)
    assert torch.equal(maps, torch.sigmoid(expected_logits))
    assert torch.equal(scores, expected_scores)
    assert torch.all(scores < 0) and torch.all(maps >= 0) and torch.all(maps <= 1)
    for index, record in enumerate(manifest["splits"]["test"]):
        result = engine.inspect(record["image"], "kolektor_surface")
        summary = result["summary"]
        assert summary["score"] == pytest.approx(float(scores[index]), rel=1e-6, abs=1e-6)
        assert summary["predicted_defective"] == bool(scores[index] >= engine.threshold)
        assert summary["threshold"] == engine.threshold
        assert summary["display_scale"]["minimum"] == 0 and summary["display_scale"]["maximum"] == 1
        assert summary["experimental_status"] == "exploratory"
        assert "segmentation" in summary["map_definition"]
        assert result["native_map"].shape == (record["original_size"][1], record["original_size"][0])


@pytest.mark.parametrize("candidate", ["control", "decision_head"])
def test_stream_native_ap_matches_bruteforce_and_stress_original_parity(checkpoints, tmp_path, monkeypatch, candidate):
    manifest_path, manifest, paths, _ = checkpoints
    other_splits = {row[key] for split, rows in manifest["splits"].items() if split != "test"
                    for row in rows for key in ("image", "mask")}
    original_open = Image.open

    def only_test(path, *args, **kwargs):
        assert str(path) not in other_splits
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", only_test)
    engine = inference.DecisionInspectionEngine(paths[candidate], "cpu")
    images, valid, transforms = prepared_batch(manifest["splits"]["test"], engine.preprocess)
    maps, scores = engine.predict_tensor(images, valid)
    all_maps, targets = [], []  # Deliberately brute-force only these two tiny fixture images.
    for index, record in enumerate(manifest["splits"]["test"]):
        all_maps.append(restore_map(maps[index, 0].numpy(), transforms[index]).ravel())
        with Image.open(record["mask"]) as image:
            targets.append((np.asarray(image) > 0).ravel())
    expected = average_precision_score(np.concatenate(targets), np.concatenate(all_maps))
    result = native.evaluate(manifest_path, paths[candidate], device="cpu", batch_size=2,
                             bootstrap_samples=10, scratch_root=tmp_path / "scratch")
    assert result["pixel_average_precision"] == pytest.approx(expected, abs=1e-12)
    assert [row["score"] for row in result["predictions"]] == scores.tolist()
    assert result["pixel_average_precision_by_category"]["kolektor_surface"]["pixels"] == sum(map(len, targets))
    assert result["experimental_status"] == "exploratory"
    assert result["score_definition"] == engine.score_definition and result["model_kind"] == engine.model_kind
    assert "scripts/ksdd2_decision_inference.py" in result["evaluation_additional_source_sha256"]
    assert "scripts/ksdd2_decision_native.py" in result["evaluation_additional_source_sha256"]
    assert not list((tmp_path / "scratch").iterdir())
    stress = inference.stress_test_checkpoint(manifest_path, paths[candidate], "cpu", 2)
    original = stress["results"]["original"]
    assert original["confusion"] == result["confusion"]
    assert original["image_auroc"] == result["image_auroc"]
    assert [row["score"] for row in original["predictions"]] == scores.tolist()
    assert set(stress["results"]) == {"original", "gaussian_blur_radius_1", "brightness_0.8", "brightness_1.2", "jpeg_quality_60"}
    assert all(row["threshold"] == engine.threshold for row in stress["results"].values())
    assert stress["model_kind"] == engine.model_kind and stress["score_definition"] == engine.score_definition
    json.dumps(result, allow_nan=False); json.dumps(stress, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("category", "screw"), ("score_definition", "top 1%"), ("model_kind", "supervised_segmentation"),
    ("config_digest", "a" * 64), ("declaration_digest", "missing"), ("parameter_counts", {}),
    ("threshold", float("nan")), ("model_selection", "test labels"),
])
def test_mismatches_fail_before_any_image_pixels(checkpoints, monkeypatch, field, value):
    manifest_path, _, paths, states = checkpoints
    checkpoint = copy.deepcopy(states["decision_head"]); checkpoint[field] = value
    torch.save(checkpoint, paths["decision_head"])
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("Invalid checkpoint reached image pixels"))
    with pytest.raises(ValueError):
        native.evaluate(manifest_path, paths["decision_head"], "cpu")
    with pytest.raises(ValueError):
        inference.stress_test_checkpoint(manifest_path, paths["decision_head"], "cpu")


def test_split_and_calibration_metadata_mismatch_before_content_hash(checkpoints, monkeypatch):
    manifest_path, _, paths, states = checkpoints
    checkpoint = copy.deepcopy(states["decision_head"])
    checkpoint["split_digest"] = "a" * 64
    checkpoint["best_validation"]["split_digest"] = "a" * 64
    checkpoint["provenance"]["split_digest"] = "a" * 64
    torch.save(checkpoint, paths["decision_head"])
    original = inference.load_manifest

    def metadata_only(path, verify_splits=()):
        assert not verify_splits, "Incompatible checkpoint read test content"
        return original(path)

    monkeypatch.setattr(inference, "load_manifest", metadata_only)
    with pytest.raises(ValueError, match="split differ"):
        native.evaluate(manifest_path, paths["decision_head"], "cpu")
    checkpoint = copy.deepcopy(states["decision_head"]); checkpoint["calibration"]["score_definition"] = "wrong"
    torch.save(checkpoint, paths["decision_head"])
    with pytest.raises(ValueError, match="Calibration threshold"):
        inference.DecisionInspectionEngine(paths["decision_head"], "cpu")


def test_tensor_inputs_category_and_benchmark_argument_validation(checkpoints, monkeypatch):
    _, manifest, paths, _ = checkpoints
    engine = inference.DecisionInspectionEngine(paths["decision_head"], "cpu")
    images, valid, _ = prepared_batch(manifest["splits"]["test"], engine.preprocess)
    for bad in (images.double(), images * float("nan"), images[..., :16], images + 2, images[:0]):
        with pytest.raises(ValueError, match="float32 BCHW"):
            engine.predict_tensor(bad, valid)
    for bad in (None, torch.zeros_like(valid), valid[..., :16], valid.float() * 2):
        with pytest.raises(ValueError, match="binary valid mask"):
            engine.predict_tensor(images, bad)
    with pytest.raises(ValueError, match="not screw"):
        engine.inspect("never read", category="screw")
    benchmark = inference.benchmark_checkpoint(paths["decision_head"], manifest["splits"]["test"][0]["image"],
                                              device="cpu", iterations=2, warmup=1)
    assert benchmark["measurements"] == 2 and benchmark["warmup_forwards"] == 1
    assert benchmark["single_image_median_ms"] > 0
    assert benchmark["model_kind"] == engine.model_kind and benchmark["score_definition"] == engine.score_definition
    with pytest.raises(ValueError, match="positive integer"):
        inference.benchmark_checkpoint("never", "never", iterations=0)
    for batch_size, bootstrap in ((0, 10), (True, 10), (2, 0), (2, 1.5)):
        with pytest.raises(ValueError, match="positive integer"):
            native.evaluate("never", "never", batch_size=batch_size, bootstrap_samples=bootstrap)


def test_native_padding_and_mask_alignment_are_preserved(tmp_path):
    image = Image.new("RGB", (8, 32), "gray")
    target = np.zeros((32, 8), dtype=np.uint8); target[10:14, 2:6] = 255
    path = tmp_path / "mask.png"; Image.fromarray(target).save(path)
    _, _, transform = prepare_image(image, {"mode": "letterbox", "height": 32, "width": 32})
    activation = np.ones((32, 32), dtype=np.float32)
    activation[:, transform["left"]:transform["left"] + 8] = target / 255
    spool = PixelSpools(tmp_path / "pixels", conditions=1)
    try:
        native._append_native(spool, {"mask": str(path), "original_size": [8, 32]}, activation, transform)
        ap, _, counts = spool.metrics()
        assert ap == 1. and counts == [{"negative": 240, "positive": 16}]
    finally:
        spool.close()
    Image.new("L", (9, 32), 0).save(path)
    spool = PixelSpools(tmp_path / "bad", conditions=1)
    try:
        with pytest.raises(ValueError, match="must align"):
            native._append_native(spool, {"mask": str(path), "original_size": [8, 32]}, activation, transform)
    finally:
        spool.close()
