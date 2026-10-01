import json

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.model import InspectionModel
from inspection.predict import predict


def test_prediction_uses_frozen_threshold_and_protects_existing_outputs(tmp_path):
    model = InspectionModel(base_channels=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = {
        "model_state": model.state_dict(), "model_config": {"base_channels": 4},
        "config": {"category": "metal_nut"}, "image_size": 32, "threshold": 0.5,
        "smoke_run": True,
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    image_path = tmp_path / "photo.png"
    Image.fromarray(np.full((48, 64, 3), 127, dtype=np.uint8)).save(image_path)
    output = tmp_path / "prediction"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = predict(checkpoint_path, image_path, output, "cpu")
        assert result["score"] == 0.5
        assert result["threshold"] == 0.5
        assert result["decision"] == "defective"  # Boundary uses >=, never a new threshold.
        assert result["category"] == "metal_nut"
        assert result["smoke_checkpoint"]
        assert len(result["checkpoint_sha256"]) == len(result["model_sha256"]) == 64
        assert result["inference_ms"] > 0
        assert json.loads((output / "prediction.json").read_text()) == result
        with Image.open(output / "heatmap.png") as heatmap:
            assert heatmap.size == (32, 32)
        with Image.open(output / "overlay.png") as overlay:
            assert overlay.size == (64, 48)
        with pytest.raises(FileExistsError):
            predict(checkpoint_path, image_path, output, "cpu")
        checkpoint["threshold"] = 0.6
        torch.save(checkpoint, checkpoint_path)
        second = predict(checkpoint_path, image_path, tmp_path / "second", "cpu")
        assert second["score"] == 0.5 and second["decision"] == "good"
    finally:
        torch.set_num_threads(previous_threads)
