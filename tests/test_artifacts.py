"""A model is usable only after its pinned release hash is verified."""
import hashlib
import io
import json

import pytest

from inspection.artifacts import load_registry, resolve_checkpoint


RELEASE = "https://github.com/numann44/industrial-inspection/releases/download/v1/model.pt"


def entry(payload, **extra):
    return {"sha256": hashlib.sha256(payload).hexdigest(), "url": RELEASE, **extra}


def test_changed_local_weights_are_rejected_without_network(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"changed weights")
    def forbidden(*args, **kwargs):
        pytest.fail("An invalid local checkpoint must not trigger a replacement download")
    monkeypatch.setattr("inspection.artifacts.urllib.request.urlopen", forbidden)
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_checkpoint(entry(b"original weights", local_path="model.pt"), tmp_path)


def test_verified_download_is_cached_and_reused(tmp_path, monkeypatch):
    payload = b"pinned model release"
    calls = []
    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)
    monkeypatch.setattr("inspection.artifacts.urllib.request.urlopen", download)
    model = entry(payload)
    checkpoint = resolve_checkpoint(model, tmp_path)
    assert checkpoint.read_bytes() == payload
    assert resolve_checkpoint(model, tmp_path) == checkpoint
    assert calls == [(RELEASE, 60)]
    assert not list(checkpoint.parent.glob("*.download"))


def test_wrong_download_checksum_leaves_no_model_or_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("inspection.artifacts.urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="Downloaded model checksum mismatch"):
        resolve_checkpoint(entry(b"expected"), tmp_path)
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.download"))


@pytest.mark.parametrize("url", [None, "https://example.com/model.pt", "http://github.com/numann44/industrial-inspection/releases/download/v1/model.pt"])
def test_unapproved_download_location_is_rejected(tmp_path, monkeypatch, url):
    def forbidden(*args, **kwargs):
        pytest.fail("Unapproved model URLs must not be opened")
    monkeypatch.setattr("inspection.artifacts.urllib.request.urlopen", forbidden)
    with pytest.raises(FileNotFoundError, match="not available"):
        resolve_checkpoint(entry(b"model", url=url), tmp_path)


def test_registry_rejects_unsupported_schema(tmp_path):
    registry = tmp_path / "models.json"
    registry.write_text(json.dumps({"schema_version": 2, "models": []}))
    with pytest.raises(ValueError, match="Unsupported model registry"):
        load_registry(registry)
