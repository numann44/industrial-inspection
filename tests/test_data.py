import json

import numpy as np
import pytest
from PIL import Image

from inspection.data import InspectionDataset, build_manifest, load_manifest


def make_dataset(tmp_path):
    root = tmp_path / "mvtec_ad"
    for index in range(20):
        path = root / "metal_nut/train/good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((16, 16, 3), index * 5, dtype=np.uint8)
        Image.fromarray(image).save(path)
    for defect, value in [("good", 201), ("scratch", 220)]:
        path = root / "metal_nut/test" / defect / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)
    mask_path = root / "metal_nut/ground_truth/scratch/000_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[3:8, 4:10] = 255
    Image.fromarray(mask).save(mask_path)
    return root


def test_original_test_sealed_and_calibration_separate(tmp_path):
    root = make_dataset(tmp_path)
    manifest, audit = build_manifest(root)
    assert {name: len(records) for name, records in manifest["splits"].items()} == {
        "train": 14, "validation": 3, "calibration": 3, "test": 2}
    non_test = [record for name, records in manifest["splits"].items() if name != "test" for record in records]
    assert all(record["label"] == 0 and record["mask"] is None for record in non_test)
    expected_test = {str(path) for path in (root / "metal_nut/test").glob("*/*.png")}
    assert {record["image"] for record in manifest["splits"]["test"]} == expected_test
    paths = [record["image"] for records in manifest["splits"].values() for record in records]
    assert len(paths) == len(set(paths))
    assert audit["decoded_pngs"] == 23
    assert audit["status"] == "passed"
    assert build_manifest(root)[0] == manifest


def test_missing_mask_rejected(tmp_path):
    root = make_dataset(tmp_path)
    (root / "metal_nut/ground_truth/scratch/000_mask.png").unlink()
    with pytest.raises(ValueError, match="Missing ground-truth mask"):
        build_manifest(root)


def test_cross_split_byte_duplicate_rejected(tmp_path):
    root = make_dataset(tmp_path)
    (root / "metal_nut/test/good/000.png").write_bytes((root / "metal_nut/train/good/000.png").read_bytes())
    with pytest.raises(ValueError, match="Identical image bytes cross"):
        build_manifest(root)


@pytest.mark.parametrize("problem", ["nonbinary", "wrongsize", "empty"])
def test_bad_masks_rejected(tmp_path, problem):
    root = make_dataset(tmp_path)
    path = root / "metal_nut/ground_truth/scratch/000_mask.png"
    if problem == "nonbinary":
        Image.fromarray(np.full((16, 16), 40, dtype=np.uint8)).save(path)
        expected = "binary grayscale"
    elif problem == "wrongsize":
        Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(path)
        expected = "dimensions differ"
    else:
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(path)
        expected = "empty mask"
    with pytest.raises(ValueError, match=expected):
        build_manifest(root)


def test_manifest_edit_detected_and_loader_binary_mask(tmp_path):
    root = make_dataset(tmp_path)
    manifest, _ = build_manifest(root)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert load_manifest(path)["digest"] == manifest["digest"]
    record = next(record for record in manifest["splits"]["test"] if record["label"])
    sample = InspectionDataset([record], image_size=32)[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["mask"].shape == (1, 32, 32)
    assert set(sample["mask"].unique().tolist()) == {0.0, 1.0}
    manifest["config"]["seed"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_manifest(path)


def test_split_digest_portable_across_roots_and_resolutions(tmp_path):
    first, _ = build_manifest(make_dataset(tmp_path / "first"), image_size=128)
    second, _ = build_manifest(make_dataset(tmp_path / "second"), image_size=256)
    assert first["split_digest"] == second["split_digest"]
    assert first["digest"] != second["digest"]


def test_training_verification_does_not_read_test_content(tmp_path):
    root = make_dataset(tmp_path)
    manifest, _ = build_manifest(root)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    (root / "metal_nut/test/good/000.png").write_bytes(b"changed test bytes")
    load_manifest(path, verify_splits=("train", "validation", "calibration"))
    with pytest.raises(ValueError, match="Audited content changed"):
        load_manifest(path, verify_splits=("test",))
