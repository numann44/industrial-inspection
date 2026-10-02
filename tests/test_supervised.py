import torch
import json
from pathlib import Path
import pytest

from inspection.supervised import BalancedBatchSampler, segmentation_loss, valid_anomaly_score


def test_balanced_batches_are_exact_and_seeded():
    records = [{"label": 0}] * 20 + [{"label": 1}] * 3
    def batches():
        return list(BalancedBatchSampler(records, 8, torch.Generator().manual_seed(42)))
    assert batches() == batches()
    for indices in batches():
        assert len(indices) == 8
        assert sum(records[index]["label"] for index in indices) == 4


def test_padding_does_not_change_loss_or_score_and_receives_no_gradient():
    logits = torch.zeros((2, 1, 12, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, :, 4:6, 3:5] = 1
    valid = torch.ones_like(logits)
    valid[:, :, :2] = 0
    original, _ = segmentation_loss(logits, target, valid)
    modified = logits.detach().clone()
    modified[:, :, :2] = 100
    altered, _ = segmentation_loss(modified, target, valid)
    assert torch.allclose(original, altered)
    assert torch.allclose(valid_anomaly_score(logits, valid), valid_anomaly_score(modified, valid))
    original.backward()
    assert torch.count_nonzero(logits.grad[:, :, :2]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 2:]) > 0


def test_raw_padding_activation_never_changes_operating_score():
    logits = torch.full((1, 1, 10, 10), -10.0)
    logits[:, :, :8] = 10
    valid = torch.zeros_like(logits)
    valid[:, :, 8:] = 1
    assert valid_anomaly_score(logits, valid).item() < 0.001


def test_cpu_resume_matches_full_run_without_reading_test_images(tmp_path):
    from test_ksdd2 import fixture_data
    from inspection.ksdd2 import build_manifest
    from inspection.supervised import train
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture_data(tmp_path / "dataset")
        manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        # Test bytes are deliberately invalid: training must never decode or hash them.
        Path(manifest["splits"]["test"][0]["image"]).write_bytes(b"not training data")
        config = {"base_channels": 2, "training_seed": 7, "batch_size": 4,
                  "epochs": 2, "patience": 15, "learning_rate": 0.001,
                  "positive_pixel_weight": 3.0, "threshold_quantile": 0.9,
                  "preprocess": {"mode": "letterbox", "height": 48, "width": 24}, "device": "cpu"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        full = train(manifest_path, config_path, tmp_path / "full")
        interrupted = train(manifest_path, config_path, tmp_path / "resumed", stop_after_epoch=1)
        assert interrupted["epoch"] == 1
        resumed = train(manifest_path, config_path, tmp_path / "resumed", resume=tmp_path / "resumed/last.pt")
        assert full["best_epoch"] == resumed["best_epoch"]
        assert full["threshold"] == resumed["threshold"]
        for key in full["model_state"]:
            assert torch.equal(full["model_state"][key], resumed["model_state"][key])
    finally:
        torch.set_num_threads(previous_threads)


def test_auto_device_resume_rejects_changed_resolved_device(tmp_path, monkeypatch):
    from test_ksdd2 import fixture_data
    from inspection.ksdd2 import build_manifest
    from inspection.supervised import train, MODEL_KIND
    from inspection.checkpointing import validate_resume
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture_data(tmp_path / "dataset")
        manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        config = {"base_channels": 2, "training_seed": 7, "batch_size": 4,
                  "epochs": 2, "patience": 15, "learning_rate": 0.001,
                  "positive_pixel_weight": 3.0, "threshold_quantile": 0.9,
                  "preprocess": {"mode": "letterbox", "height": 48, "width": 24}, "device": "auto"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        # Resolve auto to CPU without initializing or running an accelerator.
        monkeypatch.setattr("inspection.supervised.select_device", lambda requested: torch.device("cpu"))
        run = tmp_path / "run"
        train(manifest_path, config_path, run, stop_after_epoch=1)
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        assert checkpoint["config"]["device"] == "auto"
        assert checkpoint["device"] == "cpu"
        with pytest.raises(ValueError, match="resolved device differs"):
            validate_resume(checkpoint, checkpoint["config"], manifest["split_digest"],
                            MODEL_KIND, actual_device=torch.device("mps"))
        # Verify train passes its resolved device into the common guard, rather
        # than merely recording it. A checkpoint produced on another device is
        # rejected while the test itself continues to use CPU only.
        checkpoint["device"] = "mps"
        torch.save(checkpoint, run / "last.pt")
        with pytest.raises(ValueError, match="resolved device differs"):
            train(manifest_path, config_path, run, resume=run / "last.pt")
    finally:
        torch.set_num_threads(previous_threads)
