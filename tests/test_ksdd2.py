import importlib.util
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from inspection.ksdd2 import build_manifest, load_manifest, grouped_stratified_split


def write_pair(root, split, index, label, color):
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    array = np.full((48, 20, 3), color, dtype=np.uint8)
    array[8:15, 3:9] = color // 2
    mask = np.zeros((48, 20), dtype=np.uint8)
    if label:
        mask[20:23, 8:10] = 255
    Image.fromarray(array).save(directory / f"{index}.png")
    Image.fromarray(mask).save(directory / f"{index}_GT.png")


def fixture_data(root):
    for i in range(12):
        write_pair(root, "train", 10000 + i, i % 2, 20 + i * 16)
    write_pair(root, "test", 20000, 0, 221)
    write_pair(root, "test", 20001, 1, 240)


def test_deterministic_group_split_masks_and_integrity(tmp_path):
    fixture_data(tmp_path)
    first, audit = build_manifest(tmp_path, strict_counts=False)
    second, _ = build_manifest(tmp_path, strict_counts=False)
    assert first["split_digest"] == second["split_digest"]
    assert audit["canonical_images"] == 14
    assert audit["decoded_pngs"] == 28
    assert len(first["splits"]["test"]) == 2
    groups = [{r["group"] for r in first["splits"][name]} for name in ("train", "validation", "calibration", "test")]
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4))
    output = tmp_path / "manifest.json"
    import json
    output.write_text(json.dumps(first))
    load_manifest(output, verify_splits=("train",))
    changed = Path(first["splits"]["train"][0]["image"])
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content changed"):
        load_manifest(output, verify_splits=("train",))


def test_cross_official_split_pixel_duplicate_is_quarantined(tmp_path):
    fixture_data(tmp_path)
    original = tmp_path / "train/10000.png"
    (tmp_path / "test/20000.png").write_bytes(original.read_bytes())
    manifest, audit = build_manifest(tmp_path, strict_counts=False)
    assert "train/10000.png" in audit["quarantined_official_train_images"]
    assert len(manifest["splits"]["test"]) == 2
    assert not any(r["relative_image"] == "train/10000.png" for name in ("train", "validation", "calibration")
                   for r in manifest["splits"][name])


def test_documented_copy_pair_excluded_only_after_exact_match(tmp_path):
    fixture_data(tmp_path)
    # Rename one canonical fixture to the documented original ID.
    for suffix in ("", "_GT"):
        source = tmp_path / f"train/10000{suffix}.png"
        target = tmp_path / f"train/10301{suffix}.png"
        source.rename(target)
        copy = tmp_path / f"train/10301{suffix} (copy).png"
        copy.write_bytes(target.read_bytes())
    _, audit = build_manifest(tmp_path, strict_counts=False)
    assert len(audit["excluded_verified_copy_files"]) == 2
    (tmp_path / "train/10301 (copy).png").write_bytes(b"different")
    with pytest.raises(ValueError, match="does not match"):
        build_manifest(tmp_path, strict_counts=False)


def test_near_groups_remain_together():
    records = [{"group": f"{label}-{i // 2}", "label": label, "relative_image": f"{label}/{i}"}
               for label in (0, 1) for i in range(20)]
    splits = grouped_stratified_split(records, 42)
    membership = {}
    for name, rows in splits.items():
        for row in rows:
            assert membership.setdefault(row["group"], name) == name


def test_safe_zip_rejects_traversal_before_extracting(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_ksdd2.py"
    spec = importlib.util.spec_from_file_location("prepare_ksdd2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("../outside", "unsafe")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        module.extract_archive(archive, tmp_path / "destination")
    assert not (tmp_path / "outside").exists()
