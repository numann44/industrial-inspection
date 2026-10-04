"""Tiny CPU checks for isolated v3; never access real training/test data."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from inspection.checkpointing import atomic_json, config_digest
from inspection.data import build_manifest, load_manifest
from inspection.synthesis import foreground_mask
from inspection.validation_bank import create_bank as create_v2_bank

from experiments.study_v3.bank import (NEW_FAMILIES, create_bank, family_metrics, load_bank,
                                       validate_split_sources)
from experiments.study_v3.protocol import config, continuation_manifest, gate, immutable_json, source_hashes
from experiments.study_v3.synthesis import synthesize, boundary_warp, changed_mask
from experiments.study_v3.trainer import SyntheticDataset, train


def normal_image(index=0, size=32):
    yy, xx = np.mgrid[:size, :size]
    values = np.full((size, size, 3), 22 + index, dtype=np.uint8)
    inside = (xx - size / 2) ** 2 + (yy - size / 2) ** 2 < (size * .35) ** 2
    for channel in range(3):
        values[:, :, channel][inside] = ((xx * (channel + 2) + yy * 2 + 55 + index * 4) % 190 + 30)[inside]
    return values


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "dataset"
    category = root / "metal_nut"
    for sub in ("train/good", "test/good", "test/scratch", "ground_truth/scratch"):
        (category / sub).mkdir(parents=True)
    for index in range(12):
        Image.fromarray(normal_image(index)).save(category / "train/good" / f"{index:03d}.png")
    Image.fromarray(normal_image(20)).save(category / "test/good/000.png")
    Image.fromarray(normal_image(21)).save(category / "test/scratch/000.png")
    mask = np.zeros((32, 32), np.uint8)
    mask[12:20, 15:18] = 255
    Image.fromarray(mask).save(category / "ground_truth/scratch/000_mask.png")
    manifest, _ = build_manifest(root, ["metal_nut"], seed=42, image_size=32)
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, manifest)
    create_v2_bank(manifest_path, tmp_path / "old-bank", image_size=32)
    create_bank(manifest_path, tmp_path / "old-bank/bank.json", tmp_path / "v3-bank")
    return manifest_path, tmp_path / "v3-bank/bank.json"


def tensors():
    return [torch.from_numpy(normal_image(i)).permute(2, 0, 1).float() / 255 for i in (0, 3)]


def test_synthesis_is_deterministic_preserves_inputs_and_does_not_train_with_withheld_operator():
    image, donor = tensors()
    original, original_donor = image.clone(), donor.clone()
    for candidate in ("control", "A", "B"):
        for mode in ("appearance", "texture", "warp", "scratch", "lighting"):
            first = synthesize(image, donor, torch.Generator().manual_seed(12), candidate, mode=mode)
            second = synthesize(image, donor, torch.Generator().manual_seed(12), candidate, mode=mode)
            assert all(torch.equal(a, b) for a, b in zip(first, second))
            assert torch.equal(first[1], changed_mask(image, first[0]))
            assert 0 <= first[0].min() <= first[0].max() <= 1
    assert torch.equal(image, original) and torch.equal(donor, original_donor)
    with pytest.raises(ValueError, match="bank-only"):
        synthesize(image, donor, torch.Generator(), "A", mode="withheld/hard_translation")


def test_b_changes_only_warp_family_and_can_move_original_background_pixels():
    image, donor = tensors()
    for mode in ("texture", "appearance", "scratch", "lighting"):
        a = synthesize(image, donor, torch.Generator().manual_seed(8), "A", mode=mode)
        b = synthesize(image, donor, torch.Generator().manual_seed(8), "B", mode=mode)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
    outside = ~foreground_mask(image)
    assert any((changed_mask(image, boundary_warp(image, torch.Generator().manual_seed(seed))).bool() & outside).any()
               for seed in range(20))


def test_donors_are_training_only_and_cross_split_duplicates_are_rejected(prepared):
    manifest_path, _ = prepared
    manifest = load_manifest(manifest_path)
    cfg = {**config("A", "cpu"), "image_size": 32}
    dataset = SyntheticDataset(manifest, cfg)
    train_ids = {r["image_sha256"] for r in manifest["splits"]["train"]}
    other_ids = {r["image_sha256"] for name in ("validation", "calibration", "test") for r in manifest["splits"][name]}
    for index in range(len(dataset)):
        row = dataset[index]
        assert row["donor_sha256"] in train_ids and row["donor_sha256"] not in other_ids
        assert row["source_sha256"] != row["donor_sha256"]
    manifest["splits"]["train"][0]["image_sha256"] = manifest["splits"]["validation"][0]["image_sha256"]
    with pytest.raises(ValueError, match="cross development splits"):
        validate_split_sources(manifest)


def test_bank_retains_old_bytes_and_uses_only_validation_sources_and_donors(prepared, tmp_path):
    manifest_path, path = prepared
    manifest, bank = load_manifest(manifest_path), load_bank(path)
    old = json.loads((tmp_path / "old-bank/bank.json").read_text())
    assert bank["old_bank_digest"] == old["bank_digest"]
    assert len(bank["family_counts"]) == 12  # clean + 8 old + 3 new
    assert set(bank["family_counts"].values()) == {len(manifest["splits"]["validation"])}
    ids = {r["image_sha256"] for r in manifest["splits"]["validation"]}
    for index, row in enumerate(bank["records"]):
        assert row["source_sha256"] in ids
        assert row["donor_sha256"] is None or row["donor_sha256"] in ids
        if row["origin"] == "unchanged_v2":
            assert row["image_sha256"] == old["records"][index]["image_sha256"]
            assert row["mask_sha256"] == old["records"][index]["mask_sha256"]
    with pytest.raises(FileExistsError):
        create_bank(manifest_path, tmp_path / "old-bank/bank.json", path.parent)
    array = path.parent / bank["records"][0]["image"]
    array.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content mismatch"):
        load_bank(path)


def test_family_metric_is_equal_weight_not_pooled_image_count():
    empty = np.zeros((1, 2, 2), dtype=np.float32)
    full = np.ones_like(empty)
    records = {"clean": [(0, .1, empty, empty)],
               "old/example": [(1, .9, full, full)] * 7,
               NEW_FAMILIES[0]: [(1, .01, full, empty)]}
    bank = {"family_counts": {k: len(v) for k, v in records.items()}, "bank_digest": "bank", "split_digest": "split"}
    result = family_metrics(records, bank)
    expected = np.mean([r["harmonic_mean"] for r in result["families"].values()])
    assert result["macro_h"] == expected
    assert result["macro_h"] != pytest.approx((result["old_macro_h"] * 7 + result["new_macro_h"]) / 8)


def candidate(name, score, old=.8, **extra):
    return {"candidate": name, "macro_h": score, "old_macro_h": old,
            "bank_digest": "fixed-bank", "split_digest": "fixed-split", **extra}


def test_gate_ignores_test_scores_refuses_regression_and_caps_continuation():
    control = candidate("control", .8)
    failed = gate(control, [candidate("A", .805, test_recall=1.), candidate("B", .95, old=.7)])
    assert failed["selected"] is None
    assert continuation_manifest(failed)["continuation_jobs"] == []
    accepted = gate(control, [candidate("A", .82, test_recall=0.), candidate("B", .815, test_recall=1.)])
    assert accepted["selected"]["candidate"] == "A"
    continuation = continuation_manifest(accepted)
    assert len(continuation["continuation_jobs"]) == 6
    assert continuation["maximum_new_runs_including_screen"] == 9
    assert continuation["screen_runs"] == 3 and not continuation["executor_implemented"]
    with pytest.raises(ValueError, match="bank or split differs"):
        gate(control, [candidate("A", .9, bank_digest="other"), candidate("B", .9)])


def test_cpu_resume_matches_full_run_and_never_reads_test_images(prepared, tmp_path, monkeypatch):
    manifest_path, bank_path = prepared
    cfg = {**config("B", "cpu"), "image_size": 32, "base_channels": 2,
           "batch_size": 2, "epochs": 2, "selection_every": 1}
    frozen = source_hashes()
    declaration = {"config": cfg, "sources": frozen}
    declaration_path = tmp_path / "declaration.json"
    immutable_json(declaration_path, declaration)
    digest = config_digest(declaration)
    image_open = Image.open

    def guarded_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and "test" in Path(path).parts:
            raise AssertionError("Training opened a test image")
        return image_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", guarded_open)
    full = train(manifest_path, bank_path, cfg, tmp_path / "full", digest, expected_sources=frozen)
    paused = train(manifest_path, bank_path, cfg, tmp_path / "resumed", digest,
                   stop_after_epoch=1, expected_sources=frozen)
    assert paused["epoch"] == 1 and paused["optimizer_state"] and paused["rng_state"]["shuffle"] is not None
    resumed = train(manifest_path, bank_path, cfg, tmp_path / "resumed", digest,
                    resume=tmp_path / "resumed/last.pt", expected_sources=frozen)
    assert full["best_metric"] == resumed["best_metric"]
    assert full["threshold"] == resumed["threshold"]
    assert full["best_validation"] == resumed["best_validation"]
    for name in full["model_state"]:
        assert torch.equal(full["model_state"][name], resumed["model_state"][name])
    with pytest.raises(ValueError, match="configuration differs"):
        train(manifest_path, bank_path, {**cfg, "learning_rate": .0004}, tmp_path / "resumed", digest,
              expected_sources=frozen)
    with pytest.raises(ValueError, match="source snapshot"):
        train(manifest_path, bank_path, cfg, tmp_path / "new", digest, expected_sources={})
    with pytest.raises(ValueError, match="declaration differs"):
        train(manifest_path, bank_path, cfg, tmp_path / "resumed", "wrong", expected_sources=frozen)
    with pytest.raises(ValueError, match="Immutable record"):
        immutable_json(declaration_path, {**declaration, "changed": True})


def test_nonempty_new_run_refuses_overwrite(prepared, tmp_path):
    manifest, bank = prepared
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    cfg = {**config("A", "cpu"), "image_size": 32}
    with pytest.raises(FileExistsError, match="nonempty"):
        train(manifest, bank, cfg, output, "declaration", expected_sources=source_hashes())
    assert (output / "keep.txt").read_text() == "preserve"


def test_declaration_rejects_budget_and_gate_changes_even_if_rehashed(prepared, tmp_path, monkeypatch):
    from experiments.study_v3 import runner
    manifest_path, bank_path = prepared
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path)
    monkeypatch.setattr(runner, "require_v2_complete", lambda root: "v2-results-sha")
    declaration = {"protocol": runner.PROTOCOL, "source_files": source_hashes(),
                   "maximum_total_new_runs": 9, "selection_rule": runner.SELECTION_RULE,
                   "screen_candidates": ["control", "A", "B"],
                   "gate": {"minimum_macro_h_gain": .01, "maximum_old_macro_h_regression": .02},
                   "inputs": {"v2_results_sha256": "v2-results-sha", "manifest": str(manifest_path),
                              "manifest_file_sha256": runner.sha256(manifest_path),
                              "bank": str(bank_path), "bank_digest": bank["bank_digest"],
                              "split_digest": manifest["split_digest"]},
                   "runs": [{"config": config(name, "cpu")} for name in ("control", "A", "B")]}

    def seal(value):
        return {**value, "declaration_digest": config_digest(value)}

    assert runner.validate_declaration(seal(declaration), tmp_path)
    with pytest.raises(ValueError, match="run budget"):
        runner.validate_declaration(seal({**declaration, "maximum_total_new_runs": 10}), tmp_path)
    with pytest.raises(ValueError, match="continuation gate"):
        runner.validate_declaration(seal({**declaration, "gate": {**declaration["gate"], "minimum_macro_h_gain": 0}}), tmp_path)
    with pytest.raises(ValueError, match="Source or protocol"):
        runner.validate_declaration(seal({**declaration, "source_files": {}}), tmp_path)
