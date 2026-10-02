import json

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.baseline import (MODEL_KIND, ReconstructionBaseline, evaluate_baseline, reconstruction_score,
                                 threshold_from_normal_scores, train_baseline)
from inspection.data import build_manifest


def test_raw_reconstruction_score_keeps_local_error_without_sigmoid():
    normal = torch.zeros((1, 1, 32, 32))
    defect = normal.clone()
    defect[:, :, 2:6, 2:6] = 0.8
    assert reconstruction_score(normal).item() == 0.0
    assert reconstruction_score(defect).item() == pytest.approx(0.8)


def test_normal_calibration_threshold_is_fixed_before_test_decisions():
    normal_scores = [0.1, 0.2, 0.3, 0.4]
    threshold = threshold_from_normal_scores(normal_scores, 0.95)
    assert threshold == pytest.approx(np.quantile(normal_scores, 0.95))
    first_test = np.asarray([0.2, 0.5]) >= threshold
    second_test = np.asarray([0.9, 0.99]) >= threshold
    assert first_test.tolist() == [False, True]
    assert second_test.tolist() == [True, True]
    assert threshold_from_normal_scores(normal_scores, 0.95) == threshold
    with pytest.raises(ValueError):
        threshold_from_normal_scores([np.nan])


def test_baseline_can_learn_normal_reconstruction():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(91)
        clean = torch.zeros((2, 3, 32, 32)) + 0.2
        clean[:, :, 8:24, 8:24] = 0.75
        model = ReconstructionBaseline(base_channels=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        initial = torch.nn.functional.l1_loss(model(clean)[0], clean).item()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            noisy = (clean + torch.randn_like(clean) * 0.03).clamp(0, 1)
            reconstruction, _ = model(noisy)
            loss = torch.nn.functional.l1_loss(reconstruction, clean)
            loss.backward()
            optimizer.step()
        reconstruction, errors = model(clean)
        final = torch.nn.functional.l1_loss(reconstruction, clean).item()
        # A clear loss reduction verifies gradients and denoising targets; this
        # tiny fixture is not a claim about real defect detection performance.
        assert final < initial * 0.75, (initial, final)
        assert torch.allclose(errors, (clean - reconstruction).abs().mean(1, keepdim=True))
    finally:
        torch.set_num_threads(previous_threads)


def test_training_never_reads_test_images_and_evaluation_uses_stored_threshold(tmp_path):
    root = tmp_path / "data"
    for index in range(8):
        path = root / "metal_nut/train/good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), 20 + index * 20, dtype=np.uint8)).save(path)
    for defect, value in (("good", 210), ("scratch", 230)):
        path = root / "metal_nut/test" / defect / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(path)
    mask_path = root / "metal_nut/ground_truth/scratch/000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    Image.fromarray(mask).save(mask_path)
    manifest, _ = build_manifest(root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"category": "metal_nut", "image_size": 32, "base_channels": 4,
                                     "seed": 42, "batch_size": 2, "epochs": 1, "learning_rate": 0.001,
                                     "noise_std": 0.03, "threshold_quantile": 0.95, "device": "cpu"}))
    test_image = root / "metal_nut/test/good/000.png"
    original = test_image.read_bytes()
    test_image.write_bytes(b"test images cannot be decoded during training")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / "baseline"
        checkpoint = train_baseline(manifest_path, config_path, output)
        assert checkpoint["model_kind"] == MODEL_KIND
        assert checkpoint["calibration"]["count"] == len(manifest["splits"]["calibration"])
        test_image.write_bytes(original)
        metrics = evaluate_baseline(manifest_path, output / "checkpoint.pt", device_name="cpu")
        assert metrics["threshold"] == checkpoint["threshold"]
        assert metrics["images"] == 2
        assert metrics["pixel_metric_space"] == "original"
        assert metrics["threshold_source"] == "separate normal calibration split"
    finally:
        torch.set_num_threads(previous_threads)


def test_baseline_exact_cpu_resume_includes_noise_rng(tmp_path):
    from test_train import _resume_fixture
    _, manifest_path, config_path, config = _resume_fixture(tmp_path)
    config.pop("normal_probability")
    config["noise_std"] = 0.03
    config_path.write_text(json.dumps(config))
    complete = train_baseline(manifest_path, config_path, tmp_path / "complete", max_steps=2)
    train_baseline(manifest_path, config_path, tmp_path / "resumed", max_steps=2, stop_after_epoch=1)
    resumed = train_baseline(manifest_path, config_path, tmp_path / "resumed", max_steps=2,
                             resume=tmp_path / "resumed" / "last.pt")
    assert complete["threshold"] == resumed["threshold"]
    for key in complete["model_state"]:
        assert torch.equal(complete["model_state"][key], resumed["model_state"][key]), key
    for first, second in zip(complete["history"], resumed["history"]):
        assert first["train"] == second["train"]
