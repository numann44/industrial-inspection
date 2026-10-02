"""Training integration checks that protect the evaluation/calibration boundaries."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.data import build_manifest
from inspection.train import train


def _resume_fixture(tmp_path):
    root = tmp_path / "dataset"
    random = np.random.default_rng(9)
    for index in range(10):
        path = root / "metal_nut" / "train" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.full((32, 32, 3), 20, dtype=np.uint8)
        array[6:26, 6:26] = random.integers(90, 170, (20, 20, 3), dtype=np.uint8)
        array[13:19, 13:19] = 20
        Image.fromarray(array).save(path)
    for defect in ("good", "scratch"):
        path = root / "metal_nut" / "test" / defect / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(random.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
    mask_path = root / "metal_nut" / "ground_truth" / "scratch" / "000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    Image.fromarray(mask).save(mask_path)
    manifest, _ = build_manifest(root, image_size=32)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {"category": "metal_nut", "image_size": 32, "base_channels": 4,
              "training_seed": 43, "data_seed": 42, "batch_size": 2, "epochs": 3,
              "learning_rate": 0.001, "normal_probability": 0.25,
              "threshold_quantile": 0.90, "device": "cpu", "cpu_threads": 1}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return root, manifest_path, config_path, config


def test_training_uses_separate_calibration_without_reading_test_files(tmp_path):
    random = np.random.default_rng(42)
    category = tmp_path / "metal_nut"
    for index in range(12):
        path = category / "train" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(random.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
    for defect in ("good", "scratch"):
        path = category / "test" / defect / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(random.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    path = category / "ground_truth" / "scratch" / "000_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)
    manifest, _ = build_manifest(tmp_path, image_size=32)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "category": "metal_nut", "image_size": 32, "base_channels": 4,
        "seed": 42, "batch_size": 2, "epochs": 1, "learning_rate": 0.001,
        "normal_probability": 0.2, "threshold_quantile": 0.95, "device": "cpu",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # The audit inspected the original test set. Training must never reopen it.
    for record in manifest["splits"]["test"]:
        Path(record["image"]).unlink()
        if record["mask"]:
            Path(record["mask"]).unlink()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        checkpoint = train(manifest_path, config_path, tmp_path / "run", max_steps=1)
    finally:
        torch.set_num_threads(previous_threads)
    loaded = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=True)
    assert loaded["split_digest"] == manifest["split_digest"]
    assert loaded["manifest_digest"] == manifest["digest"]
    assert loaded["smoke_run"]
    assert loaded["threshold"] == checkpoint["threshold"]
    calibration = loaded["calibration"]
    assert calibration["split"] == "calibration"
    assert calibration["count"] == len(manifest["splits"]["calibration"])
    assert calibration["quantile_method"] == "linear"
    assert calibration["decision_rule"] == ">="
    expected = np.quantile(calibration["scores"], 0.95, method="linear")
    assert calibration["threshold"] == expected
    assert calibration["calibration_exceedances"] == sum(
        score >= expected for score in calibration["scores"]
    )
    assert calibration["score_std"] == np.std(calibration["scores"])


def test_exact_cpu_resume_restores_optimizer_and_shuffle_state(tmp_path):
    _, manifest_path, config_path, _ = _resume_fixture(tmp_path)
    complete = train(manifest_path, config_path, tmp_path / "complete", max_steps=2)
    paused = train(manifest_path, config_path, tmp_path / "resumed", max_steps=2, stop_after_epoch=1)
    assert paused["resumable"] and paused["epoch"] == 1 and paused["threshold"] is None
    resumed = train(manifest_path, config_path, tmp_path / "resumed", max_steps=2,
                    resume=tmp_path / "resumed" / "last.pt")
    assert complete["best_epoch"] == resumed["best_epoch"]
    assert complete["threshold"] == resumed["threshold"]
    for key in complete["model_state"]:
        assert torch.equal(complete["model_state"][key], resumed["model_state"][key]), key
    for first, second in zip(complete["history"], resumed["history"]):
        assert first["train"] == second["train"]
        assert first["validation"] == second["validation"]
    full_last = torch.load(tmp_path / "complete" / "last.pt", weights_only=True)
    resumed_last = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    assert torch.equal(full_last["rng_state"]["shuffle"], resumed_last["rng_state"]["shuffle"])
    for index, state in full_last["optimizer_state"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, resumed_last["optimizer_state"]["state"][index][name])


def test_resume_refuses_changed_configuration_or_split_and_nonempty_new_run(tmp_path):
    root, manifest_path, config_path, config = _resume_fixture(tmp_path)
    train(manifest_path, config_path, tmp_path / "run", stop_after_epoch=1)
    config_path.write_text(json.dumps({**config, "learning_rate": 0.005}))
    with pytest.raises(ValueError, match="configuration differs"):
        train(manifest_path, config_path, tmp_path / "run", resume=tmp_path / "run" / "last.pt")
    config_path.write_text(json.dumps(config))
    changed, _ = build_manifest(root, image_size=32, seed=43)
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="data_seed"):
        train(manifest_path, config_path, tmp_path / "run", resume=tmp_path / "run" / "last.pt")
    original, _ = build_manifest(root, image_size=32)
    manifest_path.write_text(json.dumps(original))
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    (unfinished / "history.json").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="nonempty"):
        train(manifest_path, config_path, unfinished)
    assert (unfinished / "history.json").read_text() == "do not overwrite"


def test_common_bank_selection_is_used_without_calibration_or_test_labels(tmp_path):
    from inspection.validation_bank import create_bank
    _, manifest_path, config_path, config = _resume_fixture(tmp_path)
    create_bank(manifest_path, tmp_path / "bank", image_size=32)
    config.update(epochs=1, require_bank=True, selection_every=5)
    config_path.write_text(json.dumps(config))
    checkpoint = train(manifest_path, config_path, tmp_path / "run", max_steps=1,
                       bank_path=tmp_path / "bank" / "bank.json")
    assert checkpoint["best_epoch"] == 1  # Final epoch is eligible even before interval five.
    assert checkpoint["history"][0]["challenge"]["real_test_images_read"] == 0
    assert checkpoint["best_metric"] == checkpoint["history"][0]["challenge"]["selection_score"]
    assert checkpoint["bank_digest"] == checkpoint["history"][0]["challenge"]["bank_digest"]
