import io

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.data import InspectionDataset
from inspection.engine import InspectionEngine, score_maps
from inspection.model import InspectionModel
from inspection.preprocessing import decode_image, prepare_image, prepare_mask, restore_map


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def checkpoint(tmp_path, threshold=0.5):
    model = InspectionModel(base_channels=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(), "model_config": {"base_channels": 4},
                "config": {"category": "metal_nut"}, "image_size": 32,
                "threshold": threshold}, path)
    return path


def test_engine_matches_dataset_and_retains_small_activations(tmp_path):
    path = checkpoint(tmp_path)
    data = torch.load(path, weights_only=True)
    data["model_state"]["segmentation.output.bias"].fill_(-12)
    data["threshold"] = 2e-6
    torch.save(data, path)
    photo = tmp_path / "photo.png"
    Image.new("RGB", (52, 48), (127, 10, 2)).save(photo)
    record = {"image": str(photo), "mask": None, "category": "metal_nut", "defect": "good", "label": 0}
    sample = InspectionDataset([record], 32)[0]
    engine = InspectionEngine(path, "cpu")
    maps, scores = engine.predict_tensor(sample["image"][None], sample["valid_mask"][None])
    result = engine.inspect(photo, "metal_nut")
    assert result["summary"]["score"] == float(scores[0])
    np.testing.assert_array_equal(result["raw_map"], maps.numpy()[0, 0])
    assert result["raw_map"].max() < 1 / 255
    assert np.asarray(result["heatmap"]).max() > 0
    with pytest.raises(ValueError, match="not screw"):
        engine.inspect(photo, "screw")


def test_letterbox_removes_padding_before_scoring_and_restoring():
    image = Image.new("RGB", (20, 60), "white")
    spec = {"mode": "letterbox", "height": 64, "width": 64}
    tensor, valid, transform = prepare_image(image, spec)
    mask = prepare_mask(Image.new("L", image.size, 255), spec)
    assert torch.equal(valid, mask.bool())
    anomaly = torch.ones((1, 1, 64, 64))
    anomaly[valid[None]] = 0.125
    assert score_maps(anomaly, valid[None]).item() == 0.125
    native = restore_map(anomaly.numpy()[0, 0], transform)
    np.testing.assert_allclose(native, 0.125)
    assert native.shape == (60, 20)


def test_exif_orientation_and_invalid_upload():
    image = Image.new("RGB", (24, 40), "white")
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (40, 24)
    assert decode_image(Image.new("RGBA", (20, 20))).mode == "RGB"
    assert decode_image(Image.new("L", (20, 20))).mode == "RGB"
    with pytest.raises(ValueError, match="Cannot decode"):
        decode_image(b"not an image")
    with pytest.raises(ValueError, match="10 MiB"):
        decode_image(b"x" * (10 * 1024 * 1024 + 1))


def test_unknown_schema_and_nonfinite_rejected(tmp_path):
    path = checkpoint(tmp_path)
    engine = InspectionEngine(path, "cpu")
    with pytest.raises(ValueError, match="finite"):
        engine.predict_tensor(torch.full((1, 3, 32, 32), float("nan")))
    data = torch.load(path, weights_only=True)
    data["schema_version"] = 99
    torch.save(data, path)
    with pytest.raises(ValueError, match="schema"):
        InspectionEngine(path, "cpu")
