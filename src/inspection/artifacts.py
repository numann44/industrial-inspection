"""Resolve only project-owned, checksum-pinned model release assets."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path


def load_registry(path):
    registry = json.loads(Path(path).read_text())
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), list):
        raise ValueError("Unsupported model registry")
    return registry


def resolve_checkpoint(entry, root):
    root = Path(root)
    candidates = [root / entry["local_path"]] if entry.get("local_path") else []
    cache = root / ".cache" / "inspection" / f"{entry['sha256']}.pt"
    candidates.append(cache)
    for path in candidates:
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Model checksum mismatch; refusing to load changed weights")
            return path
    url = entry.get("url")
    if not url or not url.startswith("https://github.com/numann44/industrial-inspection/releases/download/"):
        raise FileNotFoundError("This model release is not available yet. See the repository's current experiment status.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".download")
    digest, size = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    raise ValueError("Model asset exceeds the allowed download size")
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("Downloaded model checksum mismatch")
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return cache
