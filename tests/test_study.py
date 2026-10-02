"""Orchestration tests use fake workers; no training or official test files opened."""
import json
import os
from pathlib import Path

import pytest

from scripts.run_study import (
    BASELINE_KIND, MATRIX, StudyRunner, category_manifest, plan, quality_target_met,
    reconcile_queue, scheduler_lock, select_by_validation, sha256,
    validate_completed_queue,
    seed_variability,
    freeze_selection_rule,
)


def candidate(name, score, kind="joint_reconstruction_segmentation", seed=42):
    return {"checkpoint_sha256": name, "model_kind": kind, "validation_score": score,
            "split_digest": "fixed-split", "bank_digest": "fixed-bank", "seed": seed}


def test_selection_excludes_reference_and_never_uses_test_winning_seed():
    highest_test = {**candidate("seed44", 0.70, seed=44), "test_recall": 1.0}
    highest_validation = {**candidate("seed43", 0.90, seed=43), "test_recall": 0.5}
    reference = candidate("reference", 0.99, BASELINE_KIND)
    choice = select_by_validation([highest_test, reference, highest_validation])
    assert choice["selected"]["seed"] == 43
    assert not choice["test_metrics_used_for_selection"]
    assert len(choice["candidates"]) == 2


def test_validation_ties_use_cpu_latency_and_incompatible_banks_are_rejected():
    candidates = [candidate("slow", 0.8), candidate("fast", 0.8)]
    choice = select_by_validation(candidates, lambda item: {"slow": 20, "fast": 10}[item["checkpoint_sha256"]])
    assert choice["selected"]["checkpoint_sha256"] == "fast"
    assert len(choice["tie_measurements"]) == 2
    candidates[1]["bank_digest"] = "different-bank"
    with pytest.raises(ValueError, match="same validation bank"):
        select_by_validation(candidates)


def test_category_manifest_filters_audited_metadata_without_reading_pixels(tmp_path):
    from inspection.data import _digest, _split_digest
    source = {"root": str(tmp_path), "schema_version": 1,
              "config": {"seed": 42, "validation_fraction": .15, "calibration_fraction": .15,
                         "image_size": 128, "categories": ["screw", "transistor"]},
              "provenance": {"dataset": "MVTec AD"}, "splits": {}, "counts": {}}
    for split in ("train", "validation", "calibration", "test"):
        source["splits"][split] = [
            {"image": str(tmp_path / category / split / "missing.png"), "mask": None,
             "category": category, "defect": "good", "label": 0,
             "image_sha256": category + split, "mask_sha256": None, "original_size": [32, 32]}
            for category in ("screw", "transistor")]
    source["digest"], source["split_digest"] = _digest(source), _split_digest(source)
    filtered = category_manifest(source, "screw", 256)
    assert all(len(records) == 1 and records[0]["category"] == "screw" for records in filtered["splits"].values())
    assert filtered["digest"] == _digest(filtered)
    assert filtered["split_digest"] == _split_digest(filtered)
    assert filtered["provenance"]["parent_manifest_digest"] == source["digest"]
    assert source["config"]["image_size"] == 128
    assert not any(tmp_path.rglob("*.png"))


def test_scheduler_refuses_parallel_runner_and_recovers_dead_lease(tmp_path):
    lock = tmp_path / "lock.json"
    with scheduler_lock(lock):
        with pytest.raises(RuntimeError, match="Another study runner"):
            with scheduler_lock(lock):
                pytest.fail("A second runner acquired the same lease")
    assert not lock.exists()
    lock.write_text(json.dumps({"pid": 99999999}))
    with scheduler_lock(lock):
        assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert not lock.exists()


def test_running_initial_queue_and_missing_handoff_block_execution(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"state": "running", "pid": os.getpid()}))
    with pytest.raises(RuntimeError, match="initial queue"):
        reconcile_queue(queue)
    with pytest.raises(RuntimeError, match="handoff"):
        StudyRunner(tmp_path, device="cpu").run()
    assert not (tmp_path / "outputs").exists()


def test_failed_or_dead_incomplete_initial_queue_stops_automatic_handoff(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"state": "failed", "pid": 99999999}))
    with pytest.raises(RuntimeError, match="Initial queue failed"):
        reconcile_queue(queue, wait=True)
    queue.write_text(json.dumps({"state": "running", "pid": 99999999}))
    with pytest.raises(RuntimeError, match="stale or incomplete"):
        reconcile_queue(queue, wait=True)
    queue.write_text(json.dumps({"state": "complete", "pid": 99999999}))
    assert reconcile_queue(queue, wait=True)["state"] == "complete"


def test_handoff_checks_the_exact_completed_six_job_declaration():
    status = {"protocol": "v2", "state": "complete", "jobs": [
        {"name": name, "state": "complete", "config": f"configs/protocol_v2/{filename}"}
        for name, filename in MATRIX]}
    validate_completed_queue(status)
    status["jobs"][0]["config"] = "different-config.json"
    with pytest.raises(RuntimeError, match="mismatched"):
        validate_completed_queue(status)


def test_resumed_frozen_selection_rejects_tampered_choice_and_candidate_metadata(tmp_path):
    runner = StudyRunner(tmp_path, device="cpu")
    candidates = []
    for name, score in (("first", .7), ("best", .9)):
        path = tmp_path / f"{name}.pt"
        path.write_bytes(name.encode())
        candidates.append({**candidate(sha256(path), score), "checkpoint": path.name})
    choice = runner._freeze_selection("example", candidates)
    assert runner._freeze_selection("example", candidates) == choice
    path = runner.output / "selections" / "example.json"
    changed = json.loads(path.read_text())
    changed["selected"] = candidates[0]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="validation-only rule"):
        runner._freeze_selection("example", candidates)
    path.write_text(json.dumps(choice))
    candidates[0] = {**candidates[0], "validation_score": .8}
    with pytest.raises(ValueError, match="candidate metadata changed"):
        runner._freeze_selection("example", candidates)


def test_cached_evaluation_rejects_wrong_split_or_threshold_without_opening_test_files(tmp_path):
    runner = StudyRunner(tmp_path, device="cpu")
    item = {**candidate("checkpoint", .8), "threshold": .5}
    path = runner.output / "evaluations" / "example.json"
    path.parent.mkdir(parents=True)
    valid = {"checkpoint_sha256": "checkpoint", "split_digest": "fixed-split", "threshold": .5}
    path.write_text(json.dumps(valid))
    assert runner._evaluate_once(item, "example") == valid
    for changed in ({**valid, "split_digest": "other-split"}, {**valid, "threshold": .1}):
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="weights, split or frozen threshold"):
            runner._evaluate_once(item, "example")


def test_partial_final_checkpoint_replays_finalization_from_atomic_last_checkpoint(tmp_path, monkeypatch):
    import inspection.train
    runner = StudyRunner(tmp_path, device="cpu")
    run = tmp_path / "partial-run"
    run.mkdir()
    (run / "checkpoint.pt").write_bytes(b"published-before-summary")
    (run / "last.pt").write_bytes(b"last-complete-epoch")
    observed = []
    monkeypatch.setattr(inspection.train, "train", lambda *args, **kwargs: observed.append((args, kwargs)))
    runner._run_training("partial", tmp_path / "manifest.json", tmp_path / "config.json", run)
    assert observed[0][1]["resume"] == run / "last.pt"
    assert observed[0][1]["device_override"] == "cpu"
    assert (run / "checkpoint.pt").read_bytes() == b"published-before-summary"


def test_targets_require_both_rates_and_dry_plan_is_bounded():
    assert quality_target_met({"defect_recall": .9, "normal_false_alarm_rate": .1})
    assert not quality_target_met({"defect_recall": 1, "normal_false_alarm_rate": .11})
    assert not quality_target_met({"defect_recall": .89, "normal_false_alarm_rate": 0})
    assert not quality_target_met({"defect_recall": None, "normal_false_alarm_rate": 0})
    assert plan()["maximum_total_training_jobs"] == 17
    assert plan()["maximum_new_jobs_after_initial_queue"] == 11


def test_pretest_freeze_describes_the_actual_validation_supervision(tmp_path):
    runner = StudyRunner(tmp_path, device="cpu")
    path = tmp_path / "own.pt"
    path.write_bytes(b"own-frozen-checkpoint")
    own = {**candidate(sha256(path), .9), "checkpoint": path.name}
    mvtec = runner._freeze_weights("mvtec-pretest-freeze", [own], {"metal_nut": own})
    assert mvtec["selection_rule"] == plan()["selection"] == freeze_selection_rule()
    supervised = {**own, "model_kind": "supervised_segmentation"}
    ksdd2 = runner._freeze_weights("ksdd2-pretest-freeze", [supervised], {"kolektor_surface": supervised})
    assert ksdd2["selection_rule"] == freeze_selection_rule(supervised=True)
    assert "real-defect validation" in ksdd2["selection_rule"] and "synthetic" not in ksdd2["selection_rule"]
    with pytest.raises(ValueError, match="supervision task"):
        runner._freeze_weights("ksdd2-pretest-freeze", [own], {"kolektor_surface": own})


def test_seed_variability_reports_all_seeds_without_choosing_the_best_test_run():
    candidates = [candidate("first", .8, seed=42), candidate("second", .9, seed=43), candidate("third", .7, seed=44)]
    evaluations = {"first": {"defect_recall": .5}, "second": {"defect_recall": .6}, "third": {"defect_recall": 1.0}}
    result = seed_variability(candidates, evaluations)
    assert result["seeds"] == [42, 43, 44]
    assert result["metrics"]["defect_recall"]["values"] == [.5, .6, 1.0]
    assert result["metrics"]["defect_recall"]["mean"] == pytest.approx(.7)
    assert result["metrics"]["defect_recall"]["sample_standard_deviation"] > 0
    assert result["metrics"]["pixel_average_precision"]["count"] == 0


class FakeStudyRunner(StudyRunner):
    """Exercise complete study ordering and resumes without model/dataset work."""
    def __init__(self, root):
        super().__init__(root, device="cpu", bootstrap_samples=1)
        self.trained, self.evaluated = [], []

    def _run_training(self, name, manifest, config, run, bank=None, supervised=False):
        run = Path(run)
        if (run / "checkpoint.pt").exists():
            return run
        settings = json.loads(Path(config).read_text())
        seed = settings.get("training_seed", 42)
        score = .94 if seed == 43 else .92 if seed == 44 else .90
        if name in {variant for variant, _ in MATRIX}:
            score = {"joint-256": .70, "joint-128": .75, "unrestricted-256": .80,
                     "no-scratch-256": .90, "segmentation-256": .85, "reconstruction-256": .99}[name]
        settings.setdefault("model_kind", "supervised_segmentation" if supervised else "joint_reconstruction_segmentation")
        settings.setdefault("training_seed", seed)
        settings["device"] = "cpu"
        run.mkdir(parents=True)
        (run / "checkpoint.pt").write_bytes(name.encode())
        (run / "summary.json").write_text(json.dumps({"config": settings, "best_metric": score,
            "model_kind": settings["model_kind"], "best_epoch": 1, "threshold": .5}))
        self.trained.append(name)
        return run

    def _load_candidate(self, run, manifest):
        summary = json.loads((Path(run) / "summary.json").read_text())
        category = summary["config"].get("category", "kolektor_surface")
        return {"run": self.relative(run), "checkpoint": self.relative(Path(run) / "checkpoint.pt"),
                "checkpoint_sha256": sha256(Path(run) / "checkpoint.pt"), "model_kind": summary["model_kind"],
                "validation_score": summary["best_metric"], "bank_digest": f"bank-{category}",
                "split_digest": f"split-{category}", "config": summary["config"],
                "manifest": self.relative(manifest), "seed": summary["config"]["training_seed"]}

    def _category_inputs(self, category, config):
        manifest, bank = self.output / "manifests" / f"{category}.json", self.output / "banks" / category / "bank.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        bank.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"category": category}))
        bank.write_text("{}")
        return manifest, bank

    def _evaluate_once(self, item, name, selected=False, supervised=False):
        freeze = self.output / ("ksdd2-pretest-freeze.json" if supervised else "mvtec-pretest-freeze.json")
        assert freeze.exists(), "Opening tests before all weights were frozen"
        path = self.output / "evaluations" / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        # Seed 44 wins the test, but seed 43 won validation and must stay deployed.
        result = {"checkpoint_sha256": item["checkpoint_sha256"],
                  "defect_recall": 1.0 if item["seed"] == 44 else .5,
                  "normal_false_alarm_rate": 0.0 if item["seed"] == 44 else .2}
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result))
        self.evaluated.append(name)
        return result

    def _selected_analyses(self, category, item):
        return {"checkpoint_sha256": item["checkpoint_sha256"]}


def test_full_study_freezes_before_tests_triggers_fallback_and_resumes_without_retraining(tmp_path):
    configs = tmp_path / "configs" / "protocol_v2"
    configs.mkdir(parents=True)
    for variant, filename in MATRIX:
        config = {"category": "metal_nut", "training_seed": 42, "image_size": 256,
                  "model_kind": BASELINE_KIND if variant == "reconstruction-256" else "joint_reconstruction_segmentation"}
        (configs / filename).write_text(json.dumps(config))
    (tmp_path / "configs" / "ksdd2_supervised.json").write_text(json.dumps({"training_seed": 42}))
    runner = FakeStudyRunner(tmp_path)
    result = runner.run(handoff=True)
    assert len(runner.trained) == 17
    assert len(runner.evaluated) == 17
    assert result["conditional_supervised"]["triggered"]
    assert not result["any_validation_selected_model_met_dataset_target"]
    assert all(item["seed"] == 43 for item in result["pretest_selections"].values())
    assert result["conditional_supervised"]["selected"]["seed"] == 43
    assert not result["selection_used_test_metrics"]
    assert not (tmp_path / "outputs" / "mps-study.lock").exists()
    resumed = runner.run(handoff=True)
    assert len(runner.trained) == 17 and len(runner.evaluated) == 17
    assert resumed["pretest_selections"] == result["pretest_selections"]
