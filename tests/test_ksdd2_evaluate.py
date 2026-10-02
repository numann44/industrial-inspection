import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from inspection.ksdd2_evaluate import native_pixel_metrics, evaluate
from inspection.preprocessing import prepare_image


def test_native_pixel_ap_crops_padding_and_preserves_target(tmp_path):
    image = Image.new("RGB", (8, 32), "gray")
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    image.save(image_path)
    target = np.zeros((32, 8), dtype=np.uint8)
    target[10:14, 2:6] = 255
    Image.fromarray(target).save(mask_path)
    preprocess = {"mode": "letterbox", "height": 32, "width": 32}
    _, _, transform = prepare_image(image, preprocess)
    prediction = np.full((32, 32), 1.0, dtype=np.float32)  # Padding must never become target pixels.
    prediction[:, transform["left"]:transform["left"] + 8] = target / 255
    metrics = native_pixel_metrics([{"image": str(image_path), "mask": str(mask_path)}], [prediction], preprocess)
    assert metrics["pixel_average_precision"] == 1.0
    assert metrics["pixels"] == 256
    assert metrics["positive_pixels"] == 16


def test_full_wrapper_uses_frozen_threshold_and_native_geometry(tmp_path):
    from test_ksdd2 import fixture_data
    from inspection.ksdd2 import build_manifest
    from inspection.model import SegmentationOnlyModel
    from inspection.engine import InspectionEngine
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture_data(tmp_path / "dataset")
        manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        checkpoint_path = tmp_path / "checkpoint.pt"
        checkpoint = {"schema_version": 2, "model_kind": "supervised_segmentation",
                      "category": "kolektor_surface", "model_config": {"base_channels": 2},
                      "model_state": SegmentationOnlyModel(2).state_dict(), "threshold": 0.6,
                      "preprocess": {"mode": "letterbox", "height": 48, "width": 32},
                      "split_digest": manifest["split_digest"], "config": {"training_seed": 42},
                      "calibration": {"normal_count": 1}, "threshold_source": "normal calibration"}
        torch.save(checkpoint, checkpoint_path)
        metrics = evaluate(manifest_path, checkpoint_path, device="cpu", bootstrap_samples=10,
                           gallery=tmp_path / "gallery")
        assert metrics["threshold"] == 0.6
        assert metrics["images"] == 2
        assert metrics["evaluation_resolution"] == "original resolution per image"
        assert metrics["training_supervision"].startswith("real defective")
        assert "KolektorSDD2" in (tmp_path / "gallery/ATTRIBUTION.md").read_text()
        engine = InspectionEngine(checkpoint_path, device="cpu")
        original = manifest["splits"]["test"][0]
        inspected = engine.inspect(original["image"])
        assert inspected["summary"]["score"] == metrics["predictions"][0]["score"]
        with Image.open(original["image"]) as image:
            assert inspected["native_map"].shape == (image.height, image.width)
    finally:
        torch.set_num_threads(previous_threads)
