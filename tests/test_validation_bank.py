import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from inspection.data import build_manifest
from inspection.validation_bank import (SyntheticBankDataset, create_bank, evaluate_bank,
                                        harmonic_selection_score, load_bank)


def _normal_manifest(tmp_path):
    root = tmp_path / "source"
    yy, xx = np.mgrid[:64, :64]
    ring = (((xx - 32) ** 2 + (yy - 32) ** 2) < 24 ** 2) & (((xx - 32) ** 2 + (yy - 32) ** 2) > 9 ** 2)
    for index in range(12):
        image = np.full((64, 64, 3), 20, dtype=np.uint8)
        image[ring] = 120 + index * 5
        image[ring, 0] = np.clip(image[ring, 0] + (xx[ring] % 7), 0, 255)
        path = root / "metal_nut/train/good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
    for defect, value in (("good", 210), ("scratch", 230)):
        path = root / "metal_nut/test" / defect / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((64, 64, 3), value, dtype=np.uint8)).save(path)
    path = root / "metal_nut/ground_truth/scratch/000_mask.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.pad(np.full((8, 8), 255, dtype=np.uint8), ((20, 36), (20, 36)))).save(path)
    manifest, _ = build_manifest(root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, manifest


def test_bank_is_identical_immutable_and_reads_only_validation(tmp_path):
    manifest_path, manifest = _normal_manifest(tmp_path)
    for split in ("train", "calibration", "test"):
        for record in manifest["splits"][split]:
            Path(record["image"]).unlink()
            if record["mask"]:
                Path(record["mask"]).unlink()
    bank = create_bank(manifest_path, tmp_path / "first", image_size=64)
    second = create_bank(manifest_path, tmp_path / "second", image_size=64)
    assert bank["bank_digest"] == second["bank_digest"]
    assert len(bank["records"]) == len(manifest["splits"]["validation"]) * 9
    assert sum(record["label"] == 0 for record in bank["records"]) == len(manifest["splits"]["validation"])
    assert all(record["anomalous_pixels"] > 0 for record in bank["records"] if record["label"])
    assert bank["split_digest"] == manifest["split_digest"]
    with pytest.raises(FileExistsError):
        create_bank(manifest_path, tmp_path / "first", image_size=64)


def test_bank_content_tampering_and_canonical_masks(tmp_path):
    manifest_path, _ = _normal_manifest(tmp_path)
    create_bank(manifest_path, tmp_path / "bank", image_size=64)
    bank_path = tmp_path / "bank/bank.json"
    dataset = SyntheticBankDataset(bank_path, image_size=32)
    example = dataset[1]
    assert example["image"].shape == (3, 32, 32)
    assert example["canonical_mask"].shape == (1, 64, 64)
    assert example["canonical_mask"].sum() > 0
    source = bank_path.parent / dataset.records[0]["image"]
    source.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="content changed"):
        load_bank(bank_path)


def test_harmonic_ranking_and_baseline_score_semantics(tmp_path):
    assert harmonic_selection_score(1.0, 0.5) == pytest.approx(2 / 3)
    assert harmonic_selection_score(0.0, 0.0) == 0
    with pytest.raises(ValueError):
        harmonic_selection_score(np.nan, 0.5)
    manifest_path, _ = _normal_manifest(tmp_path)
    create_bank(manifest_path, tmp_path / "bank", image_size=64)

    class ConstantModel(torch.nn.Module):
        def forward(self, images):
            return images, torch.zeros_like(images[:, :1])

    result = evaluate_bank(ConstantModel(), "normal_only_denoising_reconstruction", tmp_path / "bank/bank.json",
                           torch.device("cpu"), image_size=32, batch_size=4)
    assert result["image_auroc"] == 0.5
    assert result["pixel_average_precision"] > 0
    assert result["selection_score"] == harmonic_selection_score(result["image_auroc"], result["pixel_average_precision"])
    assert result["real_test_images_read"] == 0
