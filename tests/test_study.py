"""Orchestration tests use fake workers; no training or official test files opened."""
import json
import os
from pathlib import Path

import pytest

from scripts.run_study import (
    BASELINE_KIND, MATRIX, StudyRunner, category_manifest, plan, quality_target_met,
    reconcile_queue, scheduler_lock, select_by_validation, sha256,
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


def test_targets_require_both_rates_and_dry_plan_is_bounded():
    assert quality_target_met({"defect_recall": .9, "normal_false_alarm_rate": .1})
    assert not quality_target_met({"defect_recall": 1, "normal_false_alarm_rate": .11})
    assert not quality_target_met({"defect_recall": .89, "normal_false_alarm_rate": 0})
    assert not quality_target_met({"defect_recall": None, "normal_false_alarm_rate": 0})
    assert plan()["maximum_total_training_jobs"] == 17
    assert plan()["maximum_new_jobs_after_initial_queue"] == 11


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
