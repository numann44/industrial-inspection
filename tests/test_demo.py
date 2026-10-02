"""Exercise a real registered CPU checkpoint and downloadable inspection evidence."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def registered_pilot():
    registry = json.loads((ROOT / "artifacts/models.json").read_text())
    pilots = [entry for entry in registry["models"] if entry.get("examples") and (
        (entry.get("local_path") and (ROOT / entry["local_path"]).is_file()) or
        (ROOT / ".cache/inspection" / f"{entry['sha256']}.pt").is_file())]
    if not pilots:
        pytest.skip("Run scripts/fetch_demo_models.py to enable the real-model demo checks")
    return pilots[0]


def test_real_example_cpu_inspection_and_download_contents(monkeypatch, registered_pilot):
    # AppTest's media manager is discarded after each run. Capture the same bytes
    # it registers for download, without making a network request or writing uploads.
    downloads = {}
    original = MemoryMediaFileStorage.load_and_get_id
    def capture(self, path_or_data, mimetype, kind, filename=None):
        if filename:
            downloads[filename] = (path_or_data, mimetype)
        return original(self, path_or_data, mimetype, kind, filename)
    monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", capture)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not app.exception
    model_selector = next(item for item in app.selectbox if item.label == "Inspection model")
    if model_selector.value["id"] != registered_pilot["id"]:
        model_selector.set_value(registered_pilot).run()
    assert not app.exception
    assert len(app.json) == 1
    record = json.loads(app.json[0].value)
    assert record["device"] == "cpu"
    assert record["checkpoint_sha256"] == registered_pilot["sha256"]
    assert record["category"] == registered_pilot["category"]
    assert record["predicted_defective"] == (record["score"] >= record["threshold"])
    data, mimetype = downloads["inspection.json"]
    if isinstance(data, bytes):
        data = data.decode()
    assert mimetype == "application/json"
    assert json.loads(data) == record
    overlay, mimetype = downloads["overlay.png"]
    assert mimetype == "image/png"
    with Image.open(io.BytesIO(overlay)) as image:
        assert list(image.size) == record["original_image_size"]
    heatmap, mimetype = downloads["heatmap.png"]
    assert mimetype == "image/png"
    with Image.open(io.BytesIO(heatmap)) as image:
        assert list(image.size) == record["model_image_size"]
    raw, mimetype = downloads["anomaly-map.npz"]
    with np.load(io.BytesIO(raw)) as maps:
        assert set(maps.files) == {"model", "original"}
        assert np.isfinite(maps["original"]).all()
        assert maps["original"].shape == tuple(reversed(record["original_image_size"]))
    assert {item.proto.label for item in app.get("download_button")} == {
        "Download result JSON", "Download overlay", "Download raw map", "Download heatmap"}
    assert {record["overlay"], record["heatmap"], record["raw_map"]} <= downloads.keys()
    assert "not a calibrated defect probability" in record["score_definition"]


def test_upload_mode_waits_for_image_without_inference(registered_pilot):
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not app.exception
    app.radio[0].set_value("Upload a photo").run()
    assert not app.exception
    assert len(app.file_uploader) == 1
    assert not app.json
    assert not app.get("download_button")
    assert {metric.label for metric in app.metric} >= {"Defect recall", "Normal false alarms"}


def test_missing_example_shows_error_and_retains_evidence(monkeypatch, registered_pilot):
    original = Path.read_bytes
    def missing_example(path):
        if path.parent == ROOT / "assets/examples":
            raise FileNotFoundError("Packaged example unavailable")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", missing_example)
    app = AppTest.from_file(ROOT / "app.py", default_timeout=45).run()
    assert not app.exception
    assert any("example image is unavailable" in item.value for item in app.error)
    assert not app.json
    assert {metric.label for metric in app.metric} >= {"Defect recall", "Normal false alarms"}
