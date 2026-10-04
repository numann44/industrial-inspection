"""Tiny CPU-only fixtures; no real dataset or production model is accessed."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from inspection.checkpointing import atomic_json, config_digest
from inspection.ksdd2 import build_manifest, load_manifest
from inspection.model import SegmentationOnlyModel
from inspection.preprocessing import prepare_image
from inspection.supervised import segmentation_loss
from experiments.ksdd2_robustness.data import TrainingDataset, create_bank, load_bank
from experiments.ksdd2_decision import trainer, runner
from experiments.ksdd2_decision.metrics import image_metrics, grouped_paired_bootstrap, calibrate
from experiments.ksdd2_decision.model import (DecisionModel, ContextHead, check_score_identity,
                                             identity, parameter_counts, make_model)
from experiments.ksdd2_decision.protocol import (CONDITIONS, PROTOCOL, SELECTION_RULE, CANDIDATES,
                                                GATE_RULE, config, gate, source_hashes, environment,
                                                check_identity, sha256, immutable_json)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prepared(tmp_path):
    from test_ksdd2 import fixture_data
    fixture_data(tmp_path / "dataset")
    manifest, _ = build_manifest(tmp_path / "dataset", strict_counts=False)
    path = tmp_path / "manifest.json"
    atomic_json(path, manifest)
    preprocess = {"mode": "letterbox", "height": 48, "width": 24}
    bank = tmp_path / "bank.json"
    create_bank(path, bank, preprocess)
    return path, bank, preprocess


def tiny_config(preprocess, candidate="decision_head"):
    return {**config(candidate, "cpu"), "epochs": 2, "selection_every": 1, "patience_epochs": 2,
            "base_channels": 2, "batch_size": 4, "preprocess": preprocess}


def declaration(cfg):
    value = {"config": cfg, "source_files": source_hashes(), "environment": environment()}
    return {**value, "declaration_digest": config_digest(value)}


def row(candidate, score=.8, pixel_ap=.8, **kwargs):
    return {"candidate": candidate, "pooled": {"image_pauc": score, "pixel_ap": pixel_ap},
            "conditions": {name: {"image_pauc": score, "pixel_ap": pixel_ap} for name in CONDITIONS},
            "bank_digest": "bank", "split_digest": "split", **kwargs}


def assert_equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal_tree(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_equal_tree(left, right)
    else:
        assert first == second


def test_identical_backbone_initialization_global_rng_and_augmentation_sequence(prepared):
    cfg = tiny_config(prepared[2])
    torch.manual_seed(42)
    previous = SegmentationOnlyModel(2)
    old_state = torch.get_rng_state().clone()
    for candidate in CANDIDATES:
        torch.manual_seed(42)
        model = make_model({**cfg, "candidate": candidate})
        assert_equal_tree(model.segmentation.state_dict(), previous.segmentation.state_dict())
        assert torch.equal(torch.get_rng_state(), old_state)
    records = load_manifest(prepared[0])["splits"]["train"]
    original = TrainingDataset(records, {**cfg, "candidate": "acquisition_aug"})
    for candidate in CANDIDATES:
        # This is the exact explicit policy alias used by the new trainer.
        adapted = TrainingDataset(records, {**cfg, "candidate": "acquisition_aug"})
        for epoch in (0, 3):
            original.set_epoch(epoch); adapted.set_epoch(epoch)
            for index in (1, 0, 1):
                assert_equal_tree(original[index], adapted[index])
            assert torch.equal(original.generator.get_state(), adapted.generator.get_state())


def test_classification_detaches_every_path_and_preserves_segmentation_updates():
    torch.manual_seed(42)
    control = DecisionModel("control", 2)
    torch.manual_seed(42)
    candidate = DecisionModel("decision_head", 2)
    image = torch.rand(2, 3, 48, 24)
    valid = torch.ones(2, 1, 48, 24, dtype=torch.bool)
    valid[..., :2] = False
    labels = torch.tensor([0., 1.])
    target = torch.zeros_like(valid, dtype=torch.float32)
    target[1, :, 10:15, 5:8] = 1.
    _, scores = candidate(image, valid)
    F.binary_cross_entropy_with_logits(scores, labels).backward()
    assert all(p.grad is None for p in candidate.segmentation.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in candidate.head.parameters())
    candidate.zero_grad(set_to_none=True)
    optimizers = [torch.optim.Adam(m.parameters(), lr=.0003) for m in (control, candidate)]
    for model, optimizer in zip((control, candidate), optimizers):
        logits, scores = model(image, valid)
        loss, _ = segmentation_loss(logits, target, valid)
        if model.candidate == "decision_head":
            loss = loss + F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward(); optimizer.step()
    assert_equal_tree(control.segmentation.state_dict(), candidate.segmentation.state_dict())


def test_context_head_ignores_invalid_features_logits_and_uses_raw_logit():
    head = ContextHead(16)
    features = torch.randn(2, 16, 8, 6, requires_grad=True)
    logits = torch.randn(2, 1, 64, 48, requires_grad=True)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[0, :, 8:56, 8:40] = True
    valid[1, :, 16:48, 16:32] = True
    coarse = F.interpolate(valid.float(), size=(8, 6), mode="nearest").bool()
    expected = head(features, logits, valid)
    changed_features = features.detach().masked_fill(~coarse, 1e9)
    changed_logits = logits.detach().masked_fill(~valid, -1e9)
    assert torch.equal(head(changed_features, changed_logits, valid), expected)
    expected.sum().backward()
    assert features.grad is None and logits.grad is None
    with torch.no_grad():
        head.output.weight.zero_(); head.output.bias.fill_(-7.)
    assert head(features, logits, valid).tolist() == [-7., -7.]
    with pytest.raises(ValueError, match="nonempty"):
        head(features, logits, torch.zeros_like(valid))


def test_parameter_counts_and_score_identity_reject_wrong_aggregation():
    model = DecisionModel("decision_head", 16)
    assert parameter_counts(model) == {"segmentation": 488705, "decision_head": 25747, "total": 514452}
    cfg = config("decision_head", "cpu")
    checkpoint = {"model_kind": model.model_kind, "score_definition": model.score_definition,
                  "decision_model_config": {"candidate": cfg["candidate"], "base_channels": 16, "head_seed": cfg["head_seed"]}}
    check_score_identity(checkpoint, cfg)
    for update in ({"score_definition": identity("control")[1]}, {"model_kind": identity("control")[0]},
                   {"decision_model_config": {}}):
        with pytest.raises(ValueError, match="score identity"):
            check_score_identity({**checkpoint, **update}, cfg)


def test_partial_auc_pools_conditions_and_matches_sklearn():
    labels = [0, 0, 1, 1]
    scores = np.asarray([[.1, .4, .7, 1.0], [.2, .5, .8, 1.1],
                         [.3, .6, .9, 1.2], [.35, .65, .95, 1.25]])
    pooled, conditions = image_metrics(labels, scores)
    assert all(row["image_pauc"] == 1 for row in conditions)
    assert pooled["image_pauc"] < 1
    assert pooled["image_pauc"] == roc_auc_score(np.repeat(labels, 4), scores.ravel(), max_fpr=.1)
    with pytest.raises(ValueError, match="finite"):
        image_metrics(labels, scores * np.nan)


def test_ceiling_aware_gate_and_all_condition_localization_guards():
    assert gate(row("control", .999), row("decision_head", .9992))["state"] == "awaiting_exploratory_evaluation"
    assert gate(row("control", 1.), row("decision_head", 1.))["state"] == "closed_failed_gate"
    assert gate(row("control", .9), row("decision_head", .905))["state"] == "closed_failed_gate"
    for section, metric in (("pooled", "pixel_ap"), ("jpeg_quality_60", "pixel_ap"),
                            ("brightness_1.2", "image_pauc")):
        candidate = row("decision_head", .95)
        (candidate[section] if section == "pooled" else candidate["conditions"][section])[metric] = .7
        assert gate(row("control"), candidate)["state"] == "closed_failed_gate"
    candidate = row("decision_head", .95, calibration_fpr=0., test_recall=1.)
    candidate["pooled"]["image_pauc"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        gate(row("control"), candidate)


def test_paired_bootstrap_resamples_original_groups():
    common = {"group_ids": list("abcde"), "group_labels": [0, 0, 0, 1, 1], "bank_digest": "bank"}
    control = {**common, "group_scores": [[v] * 4 for v in [.1, .3, .6, .4, .8]]}
    candidate = {**common, "group_scores": [[v] * 4 for v in [.1, .3, .4, .6, .8]]}
    result = grouped_paired_bootstrap(control, candidate, samples=20)
    assert result == grouped_paired_bootstrap(control, candidate, samples=20)
    assert result["difference"] > 0 and result["original_image_groups"] == 5 and not result["used_for_gate"]
    with pytest.raises(ValueError, match="identical"):
        grouped_paired_bootstrap(control, {**candidate, "group_ids": list("fghij")})


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_actual_cpu_interrupted_resume_and_forbidden_split_isolation(prepared, tmp_path, candidate):
    manifest_path, bank_path, preprocess = prepared
    manifest = load_manifest(manifest_path)
    for name in ("calibration", "test"):
        for record in manifest["splits"][name]:
            for key in ("image", "mask"):
                Path(record[key]).write_bytes(b"must not be opened during training")
    cfg = tiny_config(preprocess, candidate)
    frozen = declaration(cfg)
    full = trainer.train(manifest_path, bank_path, cfg, tmp_path / "full", frozen)
    first = trainer.train(manifest_path, bank_path, cfg, tmp_path / "resumed", frozen, stop_after_epoch=1)
    assert first["epoch"] == 1
    resumed = trainer.train(manifest_path, bank_path, cfg, tmp_path / "resumed", frozen, resume=tmp_path / "resumed/last.pt")
    assert full["best_validation"] == resumed["best_validation"]
    assert full["best_epoch"] == resumed["best_epoch"]
    assert full["threshold"] is None and resumed["threshold"] is None
    assert_equal_tree(full["model_state"], resumed["model_state"])
    last = [torch.load(tmp_path / name / "last.pt", map_location="cpu", weights_only=True) for name in ("full", "resumed")]
    assert_equal_tree(last[0]["optimizer_state"], last[1]["optimizer_state"])
    for key in ("torch", "shuffle", "augmentation"):
        assert torch.equal(last[0]["rng_state"][key], last[1]["rng_state"][key])
    for changed in ({**cfg, "learning_rate": .0004}, {**cfg, "candidate": "other"}):
        with pytest.raises(ValueError):
            trainer.train(manifest_path, bank_path, changed, tmp_path / "resumed", frozen)
    with pytest.raises(ValueError, match="Source changed"):
        trainer.train(manifest_path, bank_path, cfg, tmp_path / "unused", {**frozen, "source_files": {}})
    (tmp_path / "resumed/summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Completed summary differs"):
        trainer.train(manifest_path, bank_path, cfg, tmp_path / "resumed", frozen)


def test_exact_ties_keep_earliest_and_patience_counts_training_epochs(prepared, tmp_path, monkeypatch):
    cfg = {**tiny_config(prepared[2], "control"), "epochs": 25, "selection_every": 5, "patience_epochs": 15}
    monkeypatch.setattr(trainer, "evaluate", lambda *args: row("control"))
    result = trainer.train(prepared[0], prepared[1], cfg, tmp_path / "run", declaration(cfg))
    assert result["best_epoch"] == 5 and result["completed_epochs"] == 20
    assert result["stale_epochs"] == 15 and result["early_stopped"]


def test_failed_gate_forbids_all_calibration_access(monkeypatch):
    failed = gate(row("control"), row("decision_head", .7))
    monkeypatch.setattr(trainer, "load_manifest", lambda *args, **kwargs: pytest.fail("Calibration metadata/pixels read"))
    with pytest.raises(ValueError, match="Failed gate forbids"):
        trainer.calibrate_after_gate("unread", "unread", {}, "unread", {}, failed)


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_passed_gate_calibrates_score_once_and_rejects_corrupted_cache(prepared, tmp_path, monkeypatch, candidate):
    manifest_path, bank_path, preprocess = prepared
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path)
    cfg = tiny_config(preprocess, candidate)
    frozen = declaration(cfg)
    metric = {**row(candidate, .95 if candidate == "decision_head" else .8),
              "bank_digest": bank["bank_digest"], "split_digest": manifest["split_digest"]}
    monkeypatch.setattr(trainer, "evaluate", lambda *args: metric)
    screen = tmp_path / "screen"
    output = screen / "runs" / f"{candidate}-seed42"
    selected = trainer.train(manifest_path, bank_path, cfg, output, frozen)
    control = {**row("control"), "bank_digest": bank["bank_digest"], "split_digest": manifest["split_digest"],
               "checkpoint_sha256": sha256(output / "selected.pt") if candidate == "control" else "other-frozen-control"}
    new = {**row("decision_head", .95), "bank_digest": bank["bank_digest"], "split_digest": manifest["split_digest"],
           "checkpoint_sha256": sha256(output / "selected.pt") if candidate == "decision_head" else "other-frozen-candidate"}
    passed = gate(control, new)
    with pytest.raises(ValueError, match="Global gate must be recorded"):
        trainer.calibrate_after_gate(manifest_path, bank_path, cfg, output, frozen, passed)
    immutable_json(screen / "screen-result.json", passed)
    for split in ("calibration", "test"):
        for record in manifest["splits"][split]:
            Path(record["mask"]).write_bytes(b"calibration never needs masks")
            if split == "test" or record["label"]:
                Path(record["image"]).write_bytes(b"must not be read")
    calibrated = trainer.calibrate_after_gate(manifest_path, bank_path, cfg, output, frozen, passed)
    normal_records = [r for r in manifest["splits"]["calibration"] if not r["label"]]
    model = make_model(cfg); model.load_state_dict(selected["model_state"]); model.eval()
    with Image.open(normal_records[0]["image"]) as image:
        x, valid, _ = prepare_image(image.convert("RGB"), preprocess)
    with torch.no_grad():
        _, expected = model(x[None], valid[None])
    assert calibrated["calibration"]["scores"][0] == pytest.approx(float(expected[0]), abs=1e-6)
    assert calibrated["calibration"]["normal_count"] == len(normal_records)
    assert calibrated["calibration"]["score_definition"] == identity(candidate)[1]
    before = sha256(output / "checkpoint.pt")
    monkeypatch.setattr(trainer, "calibrate", lambda *args: pytest.fail("Frozen calibration was refit"))
    (output / "calibration-summary.json").unlink()
    reused = trainer.calibrate_after_gate(manifest_path, bank_path, cfg, output, frozen, passed)
    assert reused["threshold"] == calibrated["threshold"] and sha256(output / "checkpoint.pt") == before
    assert (output / "calibration-summary.json").is_file()
    bad_cases = [("normal_count", 999), ("quantile_method", "higher"), ("split", "test"), ("scores", [float("nan")]),
                 ("scores", []), ("score_definition", "wrong"), ("calibration_exceedances", -1)]
    for field, value in bad_cases:
        broken = copy.deepcopy(calibrated); broken["calibration"][field] = value
        torch.save(broken, output / "checkpoint.pt")
        with pytest.raises(ValueError):
            trainer.calibrate_after_gate(manifest_path, bank_path, cfg, output, frozen, passed)
    broken = copy.deepcopy(calibrated); broken["provenance"] = {}
    torch.save(broken, output / "checkpoint.pt")
    with pytest.raises(ValueError, match="provenance differs"):
        trainer.calibrate_after_gate(manifest_path, bank_path, cfg, output, frozen, passed)


def test_copied_bank_retains_identical_bytes_and_digest_without_other_splits(prepared, tmp_path):
    manifest, bank, _ = prepared
    for split in ("train", "calibration", "test"):
        for record in load_manifest(manifest)["splits"][split]:
            Path(record["image"]).write_bytes(b"not validation")
    destination = tmp_path / "copied/bank.json"
    copied = runner.copy_bank(bank, destination)
    assert copied["bank_digest"] == load_bank(bank)["bank_digest"]
    assert destination.read_bytes() == bank.read_bytes()
    destination.write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        runner.copy_bank(bank, destination)


def test_exact_two_run_manifest_settings_paths_and_source_guard(tmp_path):
    output = tmp_path / "outputs/screen"
    value = {"maximum_new_training_runs": 2,
             "runs": [{"config": config(name, "cpu"), "output": str(output / "runs" / f"{name}-seed42")}
                      for name in CANDIDATES]}
    assert len(runner.validate_runs(value, output)) == 2
    for modified in ({**value, "maximum_new_training_runs": 3}, {**value, "runs": value["runs"] * 2}):
        with pytest.raises(ValueError, match="Exactly two"):
            runner.validate_runs(modified, output)
    broken = copy.deepcopy(value); broken["runs"][1]["config"]["training_seed"] = 43
    with pytest.raises(ValueError, match="Run settings"):
        runner.validate_runs(broken, output)
    with pytest.raises(ValueError, match="outside frozen sources"):
        runner.validate_output(tmp_path, tmp_path / "experiments/output")
    frozen = declaration(config("control", "cpu"))
    check_identity(frozen)
    with pytest.raises(ValueError, match="Source changed"):
        check_identity({**frozen, "source_files": {}})
    with pytest.raises(ValueError, match="Environment changed"):
        check_identity({**frozen, "environment": {}})


def test_nonempty_outputs_and_manifest_identity_fail_closed(prepared, tmp_path):
    manifest, bank, preprocess = prepared
    cfg = tiny_config(preprocess)
    output = tmp_path / "run"; output.mkdir(); (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="nonempty"):
        trainer.train(manifest, bank, cfg, output, declaration(cfg))
    frozen = {"config": cfg, "inputs": {"manifest": str(manifest), "bank": str(bank),
              "manifest_file_sha256": "changed", "bank_file_sha256": sha256(bank)}}
    with pytest.raises(ValueError, match="differs from declaration"):
        trainer.validate_binding(frozen, cfg, manifest, bank, tmp_path / "new")
