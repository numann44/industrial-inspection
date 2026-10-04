"""Small CPU-only checks; no real project dataset or heavy training is used."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from sklearn.metrics import average_precision_score

from inspection.checkpointing import atomic_json, config_digest
from inspection.ksdd2 import build_manifest, load_manifest
from inspection.preprocessing import prepare_image, prepare_mask

from experiments.ksdd2_robustness.data import (TrainingDataset, ValidationDataset, acquisition_transform,
    condition_transform, create_bank, load_bank, verify_bank_membership)
from experiments.ksdd2_robustness.metrics import (PixelSpools, grouped_paired_bootstrap,
    image_metrics, positive_threshold_ap)
from experiments.ksdd2_robustness.protocol import (CONDITIONS, PROTOCOL, SELECTION_RULE,
    config, environment, gate, immutable_json, sha256, source_hashes)
from experiments.ksdd2_robustness.trainer import train, calibrate_after_gate


@pytest.fixture
def prepared(tmp_path):
    from test_ksdd2 import fixture_data
    fixture_data(tmp_path / "dataset")
    manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    preprocess = {"mode": "letterbox", "height": 48, "width": 24}
    create_bank(path, tmp_path / "bank.json", preprocess)
    return path, tmp_path / "bank.json", preprocess


def declaration(cfg):
    body = {"config": cfg, "source_files": source_hashes(), "environment": environment()}
    return {**body, "declaration_digest": config_digest(body)}


def tiny_config(preprocess, candidate="acquisition_aug"):
    return {**config(candidate, "cpu"), "epochs": 2, "selection_every": 1,
            "patience_epochs": 2, "base_channels": 2, "batch_size": 4,
            "preprocess": preprocess}


@pytest.mark.parametrize("labels,scores", [
    ([0, 1, 0, 1, 1, 0], [.8, .8, .8, .2, .2, .2]),
    ([1, 1, 1], [.2, .5, .5]),
    ([0, 0, 0], [.2, .5, .5]),
    ([0] * 100 + [1], list(np.linspace(0, 1, 100)) + [.6]),
    ([1, 0, 1, 0], [.9, .8, .7, .6]),
])
def test_exact_memmap_ap_matches_sklearn_with_ties_sparse_and_degenerate_labels(tmp_path, labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float32)
    spool = PixelSpools(tmp_path / "pixels", conditions=2)
    try:
        spool.append(scores[::2], labels[::2], 0)
        spool.append(scores[1::2], labels[1::2], 1)
        pooled, conditions, _ = spool.metrics()
        expected = float(average_precision_score(labels, scores)) if labels.any() else 0.
        assert pooled == pytest.approx(expected, rel=1e-12, abs=1e-12)
        for index in (0, 1):
            y, s = labels[index::2], scores[index::2]
            expected = float(average_precision_score(y, s)) if y.any() else 0.
            assert conditions[index] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    finally:
        spool.close()


def test_ap_nonfinite_and_invalid_targets_fail_explicitly(tmp_path):
    spool = PixelSpools(tmp_path / "pixels", conditions=1)
    try:
        for scores, labels in (([np.nan], [1]), ([np.inf], [0]), ([.2], [2]), ([.1, .2], [1])):
            with pytest.raises(ValueError, match="aligned finite"):
                spool.append(scores, labels, 0)
        assert positive_threshold_ap([np.array([], dtype=np.float32)], [np.array([.2], dtype=np.float32)]) == 0
    finally:
        spool.close()


def test_transforms_are_deterministic_native_resolution_and_preserve_mask_geometry(prepared):
    manifest_path, _, preprocess = prepared
    records = load_manifest(manifest_path)["splits"]["train"]
    cfg = tiny_config(preprocess)
    dataset = TrainingDataset(records, cfg)
    dataset.set_epoch(3)
    first = dataset[0]
    dataset.set_epoch(3)
    again = dataset[0]
    assert all(torch.equal(first[key], again[key]) for key in ("image", "mask", "valid_mask"))
    with Image.open(records[0]["mask"]) as mask:
        assert torch.equal(first["mask"], prepare_mask(mask, preprocess))
    with Image.open(records[0]["image"]) as image:
        native = image.convert("RGB")
        transformed = acquisition_transform(native, torch.Generator().manual_seed(99), {**cfg["augmentation"], "unchanged_probability": 0.})
        assert transformed.size == native.size
        unchanged = acquisition_transform(native, torch.Generator().manual_seed(99), {**cfg["augmentation"], "unchanged_probability": 1.})
        assert np.array_equal(native, unchanged)
        control = TrainingDataset(records, {**cfg, "candidate": "control"})[0]
        assert torch.equal(control["image"], prepare_image(native, preprocess)[0])


def test_bank_has_four_content_pinned_conditions_and_validation_membership_only(prepared):
    manifest_path, bank_path, _ = prepared
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path)
    verify_bank_membership(bank, manifest)
    dataset = ValidationDataset(bank_path)
    assert len(dataset) == 4 * len(manifest["splits"]["validation"])
    for group in range(bank["original_images"]):
        examples = [dataset[group * 4 + index] for index in range(4)]
        assert all(torch.equal(examples[0]["mask"], item["mask"]) for item in examples)
        assert all(torch.equal(examples[0]["valid_mask"], item["valid_mask"]) for item in examples)
        assert {item["condition_index"] for item in examples} == {0, 1, 2, 3}
    tampered = json.loads(bank_path.read_text())
    tampered["records"][0]["source"] = manifest["splits"]["train"][0]
    with pytest.raises(ValueError, match="substituted source"):
        verify_bank_membership(tampered, manifest)
    bank["records"][0]["tensor_sha256"]["image"] = "wrong"
    bank["bank_digest"] = config_digest({k: v for k, v in bank.items() if k != "bank_digest"})
    bank_path.write_text(json.dumps(bank))
    with pytest.raises(ValueError, match="tensor contents differ"):
        ValidationDataset(bank_path)[0]


def row(name, pooled=.8, condition=.8, **extra):
    return {"candidate": name, "pooled": {"harmonic_mean": pooled},
            "conditions": {c: {"harmonic_mean": condition} for c in CONDITIONS},
            "bank_digest": "bank", "split_digest": "split", **extra}


def test_pooled_score_detects_condition_shift_hidden_by_per_condition_auroc():
    labels = [0, 0, 1, 1]
    scores = np.array([[.1, .4, .7, 1.0], [.2, .5, .8, 1.1], [.3, .6, .9, 1.2], [.35, .65, .95, 1.25]])
    pooled, conditions = image_metrics(labels, scores)
    assert conditions == [1.] * 4 and pooled < 1.


def test_gate_requires_pooled_gain_and_same_checkpoint_all_condition_guards():
    control = row("control")
    assert gate(control, row("acquisition_aug", .82))["state"] == "awaiting_exploratory_evaluation"
    candidate = row("acquisition_aug", .85)
    candidate["conditions"]["jpeg_quality_60"]["harmonic_mean"] = .78
    assert gate(control, candidate)["state"] == "closed_failed_gate"
    assert gate(control, row("acquisition_aug", .805, calibration_fpr=0., test_recall=1.))["state"] == "closed_failed_gate"
    assert not gate(control, row("acquisition_aug", .82))["further_training_allowed_by_this_screen"]


def test_grouped_paired_bootstrap_keeps_variants_together_and_is_reproducible():
    labels = [0, 0, 0, 1, 1]
    common = {"group_ids": list("abcde"), "group_labels": labels, "bank_digest": "bank"}
    control = {**common, "group_scores": [[v] * 4 for v in [.1, .3, .6, .4, .8]]}
    candidate = {**common, "group_scores": [[v] * 4 for v in [.1, .3, .4, .6, .8]]}
    first = grouped_paired_bootstrap(control, candidate, samples=30, seed=5)
    assert first == grouped_paired_bootstrap(control, candidate, samples=30, seed=5)
    assert first["original_image_groups"] == 5 and not first["used_for_gate"]
    assert first["difference"] > 0


def test_cpu_resume_matches_full_augmented_weights_optimizer_and_no_calibration_or_test_reads(prepared, tmp_path):
    manifest_path, bank_path, preprocess = prepared
    manifest = load_manifest(manifest_path)
    # Deliberately invalidate every calibration/test image and mask after audit.
    # Selection must not even hash these bytes, much less train on them.
    for split in ("calibration", "test"):
        for record in manifest["splits"][split]:
            for key in ("image", "mask"):
                Path(record[key]).write_bytes(b"forbidden outside selection")
    cfg = tiny_config(preprocess)
    frozen = declaration(cfg)
    immutable_json(tmp_path / "declaration.json", frozen)
    full = train(manifest_path, bank_path, cfg, tmp_path / "full", frozen)
    interrupted = train(manifest_path, bank_path, cfg, tmp_path / "resumed", frozen, stop_after_epoch=1)
    assert interrupted["epoch"] == 1 and interrupted["rng_state"]["augmentation"] is not None
    resumed = train(manifest_path, bank_path, cfg, tmp_path / "resumed", frozen, resume=tmp_path / "resumed/last.pt")
    assert full["best_epoch"] == resumed["best_epoch"]
    assert full["best_validation"] == resumed["best_validation"]
    assert full["threshold"] is None and resumed["threshold"] is None
    for key in full["model_state"]:
        assert torch.equal(full["model_state"][key], resumed["model_state"][key])
    last_full = torch.load(tmp_path / "full/last.pt", map_location="cpu", weights_only=True)
    last_resumed = torch.load(tmp_path / "resumed/last.pt", map_location="cpu", weights_only=True)
    for index, state in last_full["optimizer_state"]["state"].items():
        assert all(torch.equal(value, last_resumed["optimizer_state"]["state"][index][key]) for key, value in state.items())
    for changed in ({**cfg, "learning_rate": .0004}, {**cfg, "candidate": "control"}):
        with pytest.raises(ValueError, match="configuration differs"):
            train(manifest_path, bank_path, changed, tmp_path / "resumed", frozen)
    with pytest.raises(ValueError, match="Source changed"):
        train(manifest_path, bank_path, cfg, tmp_path / "new", {**frozen, "source_files": {}})
    rejected = gate(row("control"), row("acquisition_aug", .7))
    with pytest.raises(ValueError, match="Failed gate forbids calibration"):
        calibrate_after_gate(manifest_path, bank_path, cfg, tmp_path / "resumed", frozen, rejected)


def test_patience_is_training_epochs_and_earliest_equal_checkpoint_wins(prepared, tmp_path, monkeypatch):
    from experiments.ksdd2_robustness import trainer
    manifest, bank, preprocess = prepared
    cfg = {**tiny_config(preprocess, "control"), "epochs": 30, "selection_every": 5, "patience_epochs": 15}
    frozen = declaration(cfg)
    monkeypatch.setattr(trainer, "evaluate", lambda *args, **kwargs: {"pooled": {"harmonic_mean": .5}})
    selected = train(manifest, bank, cfg, tmp_path / "run", frozen)
    assert selected["best_epoch"] == 5
    assert selected["completed_epochs"] == 20
    assert selected["stale_epochs"] == 15 and selected["early_stopped"]
    assert [r["epoch"] for r in selected["history"] if r["validation"]] == [5, 10, 15, 20]


def test_exactly_two_runs_and_frozen_config_are_enforced(prepared, tmp_path, monkeypatch):
    from experiments.ksdd2_robustness import runner
    manifest_path, bank_path, _ = prepared
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path)
    evidence = {"evidence": "fixed"}
    monkeypatch.setattr(runner, "prior_evidence", lambda root: evidence)
    output = (tmp_path / "outputs/screen").resolve()
    body = {"protocol": PROTOCOL, "source_files": source_hashes(), "environment": environment(),
            "maximum_new_training_runs": 2, "conditions": list(CONDITIONS), "selection_rule": SELECTION_RULE,
            "gate": {"minimum_pooled_h_gain": .01, "maximum_clean_h_regression": .01, "maximum_each_condition_h_regression": .01},
            "prior_evidence": evidence,
            "inputs": {"manifest": str(manifest_path), "manifest_file_sha256": sha256(manifest_path),
                       "bank": str(bank_path), "bank_digest": bank["bank_digest"], "split_digest": manifest["split_digest"]},
            "runs": [{"config": config(name, "cpu"), "output": str(output / "runs" / f"{name}-seed42")}
                     for name in ("control", "acquisition_aug")]}
    def seal(value):
        return {**value, "declaration_digest": config_digest(value)}
    assert runner.validate_declaration(seal(body), tmp_path, output)
    with pytest.raises(ValueError, match="budget/ranking/gate"):
        runner.validate_declaration(seal({**body, "maximum_new_training_runs": 3}), tmp_path, output)
    with pytest.raises(ValueError, match="Exactly two"):
        runner.validate_declaration(seal({**body, "runs": body["runs"] + [body["runs"][0]]}), tmp_path, output)
    with pytest.raises(ValueError, match="Environment changed"):
        runner.validate_declaration(seal({**body, "environment": {}}), tmp_path, output)
    with pytest.raises(ValueError, match="outside frozen sources"):
        runner.validate_output(tmp_path, tmp_path / "experiments/ksdd2_robustness/output")


def test_nonempty_output_is_not_overwritten(prepared, tmp_path):
    manifest, bank, preprocess = prepared
    cfg = tiny_config(preprocess)
    run = tmp_path / "run"
    run.mkdir()
    (run / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="nonempty"):
        train(manifest, bank, cfg, run, declaration(cfg))
    assert (run / "keep.txt").read_text() == "keep"


def test_passed_gate_calibrates_once_recovers_summary_and_keeps_positive_calibration_unread(prepared, tmp_path, monkeypatch):
    from experiments.ksdd2_robustness import trainer
    manifest_path, bank_path, preprocess = prepared
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path)
    cfg = tiny_config(preprocess)
    frozen = declaration(cfg)
    metric = {**row("acquisition_aug", .8, .8), "bank_digest": bank["bank_digest"],
              "split_digest": manifest["split_digest"]}
    monkeypatch.setattr(trainer, "evaluate", lambda *args, **kwargs: metric)
    screen = tmp_path / "screen"
    run = screen / "runs/acquisition_aug-seed42"
    train(manifest_path, bank_path, cfg, run, frozen)
    candidate = {**metric, "checkpoint_sha256": sha256(run / "selected.pt")}
    control = {**row("control", .78, .78), "bank_digest": bank["bank_digest"],
               "split_digest": manifest["split_digest"], "checkpoint_sha256": "other-frozen-control"}
    passed = gate(control, candidate)
    assert passed["state"] == "awaiting_exploratory_evaluation"
    with pytest.raises(ValueError, match="Global gate must be recorded"):
        calibrate_after_gate(manifest_path, bank_path, cfg, run, frozen, passed)
    immutable_json(screen / "screen-result.json", passed)
    for name in ("calibration", "test"):
        for record in manifest["splits"][name]:
            if name == "test" or record["label"] == 1:
                for key in ("image", "mask"):
                    Path(record[key]).write_bytes(b"not available to calibration")
    calibrated = calibrate_after_gate(manifest_path, bank_path, cfg, run, frozen, passed)
    assert calibrated["calibration"]["normal_count"] == sum(r["label"] == 0 for r in manifest["splits"]["calibration"])
    assert calibrated["threshold"] is not None
    checkpoint_hash = sha256(run / "checkpoint.pt")

    def refuse_second_fit(*args, **kwargs):
        pytest.fail("A fixed calibrated checkpoint was unnecessarily refit")

    monkeypatch.setattr(trainer, "calibrate", refuse_second_fit)
    (run / "calibration-summary.json").unlink()
    reused = calibrate_after_gate(manifest_path, bank_path, cfg, run, frozen, passed)
    assert reused["threshold"] == calibrated["threshold"]
    assert (run / "calibration-summary.json").is_file()
    assert sha256(run / "checkpoint.pt") == checkpoint_hash
    (run / "calibration-summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Calibrated summary differs"):
        calibrate_after_gate(manifest_path, bank_path, cfg, run, frozen, passed)
