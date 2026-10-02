"""Fetch and verify the small, project-owned models required by the demo."""
import json
from pathlib import Path

from inspection.artifacts import load_registry, resolve_checkpoint


def main():
    root = Path(__file__).resolve().parents[1]
    for entry in load_registry(root / "artifacts/models.json")["models"]:
        path = resolve_checkpoint(entry, root)
        print(json.dumps({"id": entry["id"], "sha256": entry["sha256"],
                          "verified_path": str(path.relative_to(root))}))


if __name__ == "__main__":
    main()
