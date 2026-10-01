"""Training integration checks that protect the evaluation/calibration boundaries."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inspection.data import build_manifest
from inspection.train import train


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
