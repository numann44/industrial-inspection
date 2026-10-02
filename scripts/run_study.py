"""Resumable bounded study: validation-only selection, frozen weights, honest targets.

Dry-run is the default. Execution additionally requires an explicit MPS scheduler
handoff; this runner never starts training alongside the initial six-job queue.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from inspection.checkpointing import atomic_json

SEEDS = (42, 43, 44)
CATEGORIES = ("metal_nut", "screw", "transistor")
MATRIX = (
    ("joint-256", "metal_nut_joint_256.json"),
    ("joint-128", "metal_nut_joint_128.json"),
    ("unrestricted-256", "metal_nut_unrestricted_256.json"),
    ("no-scratch-256", "metal_nut_no_scratch_256.json"),
    ("segmentation-256", "metal_nut_segmentation_256.json"),
    ("reconstruction-256", "metal_nut_reconstruction_256.json"),
)
BASELINE_KIND = "normal_only_denoising_reconstruction"
TIE_TOLERANCE = 1e-12


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quality_target_met(metrics):
    recall, false_alarm = metrics.get("defect_recall"), metrics.get("normal_false_alarm_rate")
    return bool(recall is not None and false_alarm is not None
                and math.isfinite(recall) and math.isfinite(false_alarm)
                and recall >= 0.90 and false_alarm <= 0.10)


def seed_variability(candidates, evaluations):
    """Descriptive training-seed variability, kept separate from image CIs."""
    metrics = ("image_auroc", "pixel_average_precision", "defect_recall", "normal_false_alarm_rate")
    result = {"seeds": [item["seed"] for item in candidates], "metrics": {},
              "scope": "same audited split and configuration; variation across model training seeds, not a confidence interval"}
    for name in metrics:
        values = [evaluations[item["checkpoint_sha256"]].get(name) for item in candidates]
        finite = [value for value in values if value is not None and math.isfinite(value)]
        result["metrics"][name] = {"values": values, "count": len(finite),
                                  "mean": statistics.mean(finite) if finite else None,
                                  "sample_standard_deviation": statistics.stdev(finite) if len(finite) > 1 else None,
                                  "minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None}
    return result


def select_by_validation(candidates, latency_measure=None, exclude_baseline=True):
    """No test-field reads; exact numerical ties use predeclared CPU latency."""
    candidates = [dict(item) for item in candidates
                  if not exclude_baseline or item["model_kind"] != BASELINE_KIND]
    if not candidates:
        raise ValueError("No eligible validation candidates")
    for item in candidates:
        score = item["validation_score"]
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Candidate validation harmonic score must be finite in [0,1]")
    banks = {item.get("bank_digest") for item in candidates}
    splits = {item["split_digest"] for item in candidates}
    if len(banks) != 1 or len(splits) != 1:
        raise ValueError("Candidates must share the same validation bank and audited split")
    best_score = max(item["validation_score"] for item in candidates)
    tied = [item for item in candidates if abs(item["validation_score"] - best_score) <= TIE_TOLERANCE]
    measurements = []
    if len(tied) > 1:
        if latency_measure is None:
            raise ValueError("A CPU latency measurement is required for tied validation scores")
        for item in tied:
            latency = latency_measure(item)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError("Tie-break latency must be positive and finite")
            measurements.append({"checkpoint_sha256": item["checkpoint_sha256"], "cpu_p50_ms": latency})
        latencies = {item["checkpoint_sha256"]: item["cpu_p50_ms"] for item in measurements}
        tied.sort(key=lambda item: (latencies[item["checkpoint_sha256"]], item["checkpoint_sha256"]))
    return {"selected": tied[0], "candidates": candidates,
            "rule": "maximum validation harmonic mean; ties within 1e-12 use idle CPU p50, then checkpoint SHA256",
            "tie_tolerance": TIE_TOLERANCE, "tie_measurements": measurements,
            "test_metrics_used_for_selection": False}


def category_manifest(source, category, image_size):
    """Filter already-audited metadata only; never reopen test images or masks."""
    from inspection.data import _digest, _split_digest
    result = copy.deepcopy(source)
    result["splits"] = {name: [record for record in records if record["category"] == category]
                        for name, records in result["splits"].items()}
    if any(not result["splits"].get(name) for name in ("train", "validation", "calibration", "test")):
        raise ValueError(f"Audited manifest lacks required {category} partitions")
    result["config"]["categories"] = [category]
    result["config"]["image_size"] = image_size
    result["provenance"]["parent_manifest_digest"] = source["digest"]
    result["counts"] = {name: {"total": len(records), "by_category_defect": dict(sorted(Counter(
        f"{record['category']}/{record['defect']}" for record in records).items()))}
        for name, records in result["splits"].items()}
    result["digest"], result["split_digest"] = _digest(result), _split_digest(result)
    return result


def ensure_immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Declared artifact changed: {path}")
    else:
        atomic_json(path, value)


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip().startswith("Z"):
            return False
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def reconcile_queue(path, wait=False, sleep=time.sleep, on_status=None):
    """Wait or refuse while the existing queue owns MPS; stale jobs can resume."""
    path = Path(path)
    while path.exists():
        status = json.loads(path.read_text())
        if on_status is not None:
            on_status(status)
        alive = process_alive(status.get("pid"))
        if status.get("state") == "failed":
            raise RuntimeError("Initial queue failed; automatic scheduler handoff is stopped")
        if status.get("state") == "complete" and not alive:
            return status
        if not alive:
            raise RuntimeError("Initial queue status is stale or incomplete; verify its runs before scheduler handoff")
        if not wait:
            raise RuntimeError("The initial queue still owns MPS; wait for completion and obtain scheduler handoff")
        sleep(30)
    return None


def validate_completed_queue(status):
    if status is None:
        raise RuntimeError("The authorized initial queue status is missing")
    expected = {name: f"configs/protocol_v2/{filename}" for name, filename in MATRIX}
    jobs = status.get("jobs", [])
    if status.get("protocol") != "v2" or status.get("state") != "complete" or len(jobs) != len(expected):
        raise RuntimeError("Initial queue completion metadata differs from the declared six-job study")
    if {item.get("name") for item in jobs} != set(expected):
        raise RuntimeError("Initial queue candidate names differ")
    if any(item.get("state") != "complete" or item.get("config") != expected[item["name"]] for item in jobs):
        raise RuntimeError("Initial queue contains incomplete or mismatched jobs")


@contextlib.contextmanager
def scheduler_lock(path):
    """Project-wide outer-runner lease; stale dead-process leases are recoverable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize stale-lease recovery as well as creation. Without this guard,
    # two recoverers could both observe a dead PID and unlink each other's new
    # leases. Keep the guard inode persistent to avoid an unlink/lock race.
    with path.with_name(path.name + ".guard").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            if path.exists():
                try:
                    owner = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError("Existing scheduler lease cannot be verified; do not start a second runner") from error
                if process_alive(owner.get("pid")):
                    raise RuntimeError("Another study runner currently owns the project scheduler")
                path.unlink()
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(handle, "w") as stream:
                json.dump({"pid": os.getpid(), "created_at": timestamp()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def plan():
    return {
        "protocol": "study-v2", "training_seeds": list(SEEDS), "data_seed": 42,
        "categories": list(CATEGORIES), "calibration_quantile": 0.90,
        "selection": "common synthetic validation harmonic score; never choose a test-winning seed",
        "existing_matrix_jobs": 6, "additional_metal_nut_seeds": 2,
        "additional_category_jobs": 6, "maximum_conditional_supervised_jobs": 3,
        "maximum_total_training_jobs": 17, "maximum_new_jobs_after_initial_queue": 11,
        "held_out_evaluations": "14 MVTec checkpoints, plus 3 conditional KSDD2 checkpoints",
        "quality_target": {"defect_recall_minimum": 0.90, "normal_false_alarm_rate_maximum": 0.10},
        "conditional_fallback": "KSDD2 three-seed supervised training only if no validation-selected MVTec category meets the frozen-test target",
        "weights_frozen_before_tests": True, "failed_targets_are_reported": True,
        "metal_nut_status": "exploratory: this category's test set was inspected in earlier pilots",
    }


def freeze_selection_rule(supervised=False):
    if supervised:
        return ("real-defect validation harmonic mean of image AUROC and model-space pixel AP; "
                "ties within 1e-12 use idle CPU p50, then checkpoint SHA256; never choose a test-winning seed")
    return plan()["selection"]


class StudyRunner:
    def __init__(self, root, output="outputs/study-v2", device="mps", bootstrap_samples=1000):
        self.root = Path(root).resolve()
        self.output = self.root / output
        self.device, self.bootstrap_samples = device, bootstrap_samples
        self.status = {"protocol": "study-v2", "state": "initialized", "steps": {}, "plan": plan()}

    def path(self, value):
        return self.root / value

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def _save_status(self):
        atomic_json(self.output / "status.json", self.status)

    def _step(self, name, action):
        self.status["steps"][name] = {"state": "running", "started_at": timestamp()}
        self.status["current_step"] = name
        self._save_status()
        print(json.dumps({"study_step": name, "state": "running"}), flush=True)
        try:
            result = action()
        except BaseException as error:
            self.status["steps"][name].update(state="failed", error=str(error), finished_at=timestamp())
            self.status["state"] = "failed"
            self._save_status()
            raise
        self.status["steps"][name].update(state="complete", finished_at=timestamp())
        self._save_status()
        return result

    def _load_candidate(self, run, manifest):
        import torch
        from inspection.evaluate import load_evaluation_manifest
        run = Path(run)
        summary = json.loads((run / "summary.json").read_text())
        checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        for key in ("model_kind", "best_metric", "best_epoch", "split_digest", "config", "threshold", "bank_digest"):
            if summary.get(key) != checkpoint.get(key):
                raise ValueError(f"Completed checkpoint and summary disagree: {key}")
        source = self.root / "src" / "inspection"
        source_maps = (checkpoint["provenance"]["source_files"],
                       checkpoint["provenance"].get("supervised_source_files", {}))
        for name, digest in {name: digest for mapping in source_maps for name, digest in mapping.items()}.items():
            if not (source / name).is_file() or sha256(source / name) != digest:
                raise ValueError(f"Completed run training source changed: {name}")
        for name, version in checkpoint["provenance"]["packages"].items():
            if version is not None and importlib.metadata.version(name) != version:
                raise ValueError(f"Completed run package version changed: {name}")
        metadata = load_evaluation_manifest(manifest, verify_splits=())
        if summary.get("split_digest") != metadata["split_digest"]:
            raise ValueError("Completed checkpoint summary belongs to a different audited split")
        return {"run": self.relative(run), "checkpoint": self.relative(run / "checkpoint.pt"),
                "checkpoint_sha256": sha256(run / "checkpoint.pt"), "model_kind": summary["model_kind"],
                "validation_score": summary["best_metric"], "best_epoch": summary["best_epoch"],
                "bank_digest": summary.get("bank_digest"), "split_digest": summary["split_digest"],
                "config_digest": summary["provenance"]["config_digest"],
                "config": summary["config"], "manifest": self.relative(manifest),
                "threshold": summary["threshold"], "seed": summary["config"].get("training_seed", 42)}

    def _cpu_tie_latency(self, candidate):
        import torch
        from inspection.data import load_manifest
        from inspection.evaluate import benchmark_checkpoint
        metadata = json.loads(self.path(candidate["manifest"]).read_text())
        if metadata.get("dataset") == "KolektorSDD2":
            from inspection.ksdd2 import load_manifest as loader
        else:
            loader = load_manifest
        manifest = loader(self.path(candidate["manifest"]), verify_splits=("validation",))
        image = next(record["image"] for record in manifest["splits"]["validation"] if record["label"] == 0)
        torch.set_num_threads(1)
        result = benchmark_checkpoint(self.path(candidate["checkpoint"]), Path(image), "cpu")
        return result["single_image_median_ms"]

    def _freeze_selection(self, name, candidates):
        path = self.output / "selections" / f"{name}.json"
        if path.exists():
            choice = json.loads(path.read_text())
            eligible = [item for item in candidates if item["model_kind"] != BASELINE_KIND]
            if choice["candidates"] != eligible:
                raise ValueError("Frozen selection candidate metadata changed")
            measured = {item["checkpoint_sha256"]: item["cpu_p50_ms"] for item in choice["tie_measurements"]}
            expected = select_by_validation(candidates, lambda item: measured[item["checkpoint_sha256"]])
            if any(choice[key] != expected[key] for key in expected):
                raise ValueError("Frozen selection decision does not match the declared validation-only rule")
            for item in choice["candidates"]:
                if sha256(self.path(item["checkpoint"])) != item["checkpoint_sha256"]:
                    raise ValueError("Checkpoint changed after validation-only selection")
            return choice
        choice = select_by_validation(candidates, self._cpu_tie_latency)
        choice["frozen_at_utc"] = timestamp()
        atomic_json(path, choice)
        return choice

    def _run_training(self, name, manifest, config, run, bank=None, supervised=False):
        from inspection.train import _normalized_config, train
        manifest, config, run = map(Path, (manifest, config, run))
        # A crash can occur between publishing the inference checkpoint and
        # its summary. Replay finalization from last.pt for an incomplete pair;
        # completed pairs are never silently replaced.
        if (run / "checkpoint.pt").exists() and (run / "summary.json").exists():
            result = json.loads((run / "summary.json").read_text())
            expected = json.loads(config.read_text())
            expected["device"] = self.device
            if not supervised:
                expected = _normalized_config(expected)
            if result["config"] != expected:
                raise ValueError(f"Completed run configuration differs from the declared job: {name}")
            return run
        resume = run / "last.pt" if (run / "last.pt").exists() else None
        log = self.output / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", buffering=1) as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            if supervised:
                from inspection.supervised import train as train_supervised
                train_supervised(manifest, config, run, resume=resume, device_override=self.device)
            else:
                train(manifest, config, run, resume=resume, bank_path=bank, device_override=self.device)
        return run

    def _write_config(self, name, config):
        path = self.output / "configs" / f"{name}.json"
        ensure_immutable_json(path, config)
        return path

    def _category_inputs(self, category, config):
        from inspection.data import load_manifest
        from inspection.validation_bank import create_bank, load_bank
        source = load_manifest(self.path("data/all_categories_manifest.json"))
        manifest = category_manifest(source, category, config["image_size"])
        manifest_path = self.output / "manifests" / f"{category}.json"
        ensure_immutable_json(manifest_path, manifest)
        bank_dir = self.output / "banks" / category
        bank_path = bank_dir / "bank.json"
        if not bank_path.exists():
            if bank_dir.exists():
                raise ValueError("An incomplete final bank directory exists; preserve and review it before resuming")
            staging = bank_dir.with_name(f".{category}-building")
            # Only this runner's disposable, uncommitted normal-image bank is
            # regenerated after interruption; final bank artifacts stay immutable.
            if staging.exists():
                shutil.rmtree(staging)
            create_bank(manifest_path, staging, seed=1_000_042, image_size=256)
            load_bank(staging / "bank.json")
            staging.rename(bank_dir)
        bank = load_bank(bank_path)
        if bank["split_digest"] != manifest["split_digest"]:
            raise ValueError("Category bank split differs")
        return manifest_path, bank_path

    def _freeze_weights(self, name, candidates, selections):
        if not candidates or name not in {"mvtec-pretest-freeze", "ksdd2-pretest-freeze"}:
            raise ValueError("A declared task and nonempty candidate set are required for freezing weights")
        supervised = name == "ksdd2-pretest-freeze"
        if any((item["model_kind"] == "supervised_segmentation") != supervised for item in candidates):
            raise ValueError("Frozen model kinds differ from the declared supervision task")
        value = {"protocol": "study-v2", "checkpoints": candidates, "deployments": selections,
                 "selection_rule": freeze_selection_rule(supervised), "test_selection": False}
        ensure_immutable_json(self.output / f"{name}.json", value)
        for candidate in candidates:
            if sha256(self.path(candidate["checkpoint"])) != candidate["checkpoint_sha256"]:
                raise ValueError("Weights changed after pre-test freeze")
        return value

    def _evaluate_once(self, candidate, name, selected=False, supervised=False):
        from inspection.evaluate import evaluate_checkpoint
        path = self.output / "evaluations" / f"{name}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if (result["checkpoint_sha256"] != candidate["checkpoint_sha256"]
                    or result.get("split_digest") != candidate["split_digest"]
                    or result.get("threshold") != candidate["threshold"]):
                raise ValueError("Cached evaluation belongs to different weights, split or frozen threshold")
            return result
        category = candidate["config"].get("category", "kolektor_surface")
        gallery = self.output / "galleries" / name if selected else None
        if supervised:
            from inspection.ksdd2_evaluate import evaluate
            result = evaluate(self.path(candidate["manifest"]), self.path(candidate["checkpoint"]),
                              device=self.device, gallery=gallery, bootstrap_samples=self.bootstrap_samples)
        else:
            result = evaluate_checkpoint(self.path(candidate["manifest"]), self.path(candidate["checkpoint"]),
                                         device_name=self.device, overlay_directory=gallery,
                                         bootstrap_samples=self.bootstrap_samples,
                                         experimental_status="exploratory" if category == "metal_nut" else "frozen-held-out")
        result["measured_quality_target_met"] = quality_target_met(result)
        result["quality_target_scope"] = "dataset test point estimates; no production-line reliability claim"
        atomic_json(path, result)
        return result

    def _selected_analyses(self, category, candidate):
        import torch
        from inspection.evaluate import benchmark_checkpoint, load_evaluation_manifest, stress_test_checkpoint
        path = self.output / "analyses" / f"{category}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if (result["checkpoint_sha256"] != candidate["checkpoint_sha256"]
                    or result["stress"].get("split_digest") != candidate["split_digest"]
                    or result["stress"].get("threshold") != candidate["threshold"]):
                raise ValueError("Analysis weights, split or frozen threshold changed")
            return result
        manifest = load_evaluation_manifest(self.path(candidate["manifest"]), verify_splits=("calibration",))
        image = next(record["image"] for record in manifest["splits"]["calibration"] if record["label"] == 0)
        torch.set_num_threads(1)
        devices = ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",)
        benchmarks = {device: benchmark_checkpoint(self.path(candidate["checkpoint"]), Path(image), device)
                      for device in devices}
        result = {"checkpoint_sha256": candidate["checkpoint_sha256"],
                  "stress": stress_test_checkpoint(self.path(candidate["manifest"]), self.path(candidate["checkpoint"]), self.device),
                  "latency": benchmarks, "latency_input": "first original normal calibration image",
                  "latency_system_state": "serialized scheduler; caller must keep other compute idle"}
        atomic_json(path, result)
        return result

    def run(self, handoff=False, wait_existing=False):
        if not handoff:
            raise RuntimeError("Execution requires explicit MPS scheduler handoff")
        reconcile_queue(self.path("outputs/protocol-v2-queue.json"), wait_existing)
        with scheduler_lock(self.path("outputs/mps-study.lock")):
            try:
                return self._run()
            except BaseException as error:
                self.status.update(state="failed", error=str(error), failed_at=timestamp())
                self._save_status()
                raise

    def _run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        status_path = self.output / "status.json"
        if status_path.exists():
            self.status = json.loads(status_path.read_text())
        self.status.update(state="running", pid=os.getpid(), device=self.device)
        ensure_immutable_json(self.output / "declaration.json", {**plan(), "runner_sha256": sha256(__file__)})
        self._save_status()
        initial_candidates = []
        manifest = self.path("data/manifest.json")
        bank = self.path("data/validation-bank-v2/bank.json")
        for variant, filename in MATRIX:
            run = self.path(f"runs/protocol-v2-{variant}-seed42")
            self._step(f"initial-{variant}", lambda v=variant, f=filename, r=run: self._run_training(
                v, manifest, self.path(f"configs/protocol_v2/{f}"), r, bank))
            initial_candidates.append(self._load_candidate(run, manifest))
        variant = self._freeze_selection("main-variant", initial_candidates)["selected"]
        template = variant["config"]
        category_candidates = {"metal_nut": [variant]}
        for seed in (43, 44):
            name = f"metal_nut-seed{seed}"
            config = self._write_config(name, {**template, "training_seed": seed, "device": "auto"})
            run = self.path(f"runs/study-v2-{name}")
            self._step(name, lambda n=name, c=config, r=run: self._run_training(n, manifest, c, r, bank))
            category_candidates["metal_nut"].append(self._load_candidate(run, manifest))
        for category in ("screw", "transistor"):
            category_config = {**template, "category": category, "device": "auto"}
            category_manifest_path, category_bank = self._category_inputs(category, category_config)
            category_candidates[category] = []
            for seed in SEEDS:
                name = f"{category}-seed{seed}"
                config = self._write_config(name, {**category_config, "training_seed": seed})
                run = self.path(f"runs/study-v2-{name}")
                self._step(name, lambda n=name, c=config, r=run, m=category_manifest_path, b=category_bank:
                           self._run_training(n, m, c, r, b))
                category_candidates[category].append(self._load_candidate(run, category_manifest_path))
        selections = {category: self._freeze_selection(f"deployment-{category}", candidates)["selected"]
                      for category, candidates in category_candidates.items()}
        all_candidates = {item["checkpoint_sha256"]: item for item in initial_candidates}
        for candidates in category_candidates.values():
            all_candidates.update({item["checkpoint_sha256"]: item for item in candidates})
        self._freeze_weights("mvtec-pretest-freeze", list(all_candidates.values()), selections)
        evaluations = {}
        selected_hashes = {item["checkpoint_sha256"] for item in selections.values()}
        for candidate in all_candidates.values():
            name = Path(candidate["run"]).name
            evaluations[candidate["checkpoint_sha256"]] = self._step(f"evaluate-{name}",
                lambda c=candidate, n=name: self._evaluate_once(c, n, c["checkpoint_sha256"] in selected_hashes))
        selected_metrics = {category: evaluations[candidate["checkpoint_sha256"]]
                            for category, candidate in selections.items()}
        for category, candidate in selections.items():
            self._step(f"analyze-{category}", lambda c=category, item=candidate: self._selected_analyses(c, item))
        mvtec_pass = {category: quality_target_met(metrics) for category, metrics in selected_metrics.items()}
        fallback = {"triggered": not any(mvtec_pass.values()), "reason": "no validation-selected MVTec category met both frozen-test targets"}
        if fallback["triggered"]:
            supervised_candidates = []
            ksdd_manifest = self.path("data/ksdd2-manifest.json")
            ksdd_template = json.loads(self.path("configs/ksdd2_supervised.json").read_text())
            for seed in SEEDS:
                name = f"ksdd2-seed{seed}"
                config = self._write_config(name, {**ksdd_template, "training_seed": seed})
                run = self.path(f"runs/study-v2-{name}")
                self._step(name, lambda n=name, c=config, r=run: self._run_training(n, ksdd_manifest, c, r, supervised=True))
                supervised_candidates.append(self._load_candidate(run, ksdd_manifest))
            supervised_choice = self._freeze_selection("deployment-ksdd2", supervised_candidates)["selected"]
            self._freeze_weights("ksdd2-pretest-freeze", supervised_candidates, {"kolektor_surface": supervised_choice})
            fallback_evaluations = {}
            for candidate in supervised_candidates:
                name = Path(candidate["run"]).name
                result = self._step(f"evaluate-{name}", lambda c=candidate, n=name:
                                    self._evaluate_once(c, n, c["checkpoint_sha256"] == supervised_choice["checkpoint_sha256"], supervised=True))
                fallback_evaluations[candidate["checkpoint_sha256"]] = result
            self._step("analyze-ksdd2", lambda: self._selected_analyses("kolektor_surface", supervised_choice))
            fallback.update(selected=supervised_choice, evaluations=fallback_evaluations,
                            training_seed_variability=seed_variability(supervised_candidates, fallback_evaluations),
                            measured_quality_target_met=quality_target_met(fallback_evaluations[supervised_choice["checkpoint_sha256"]]))
        summary = {"protocol": "study-v2", "pretest_selections": selections, "mvtec_target_met": mvtec_pass,
                   "mvtec_selected_metrics": selected_metrics, "all_mvtec_evaluations": evaluations,
                   "mvtec_training_seed_variability": {category: seed_variability(candidates, evaluations)
                                                       for category, candidates in category_candidates.items()},
                   "conditional_supervised": fallback,
                   "any_validation_selected_model_met_dataset_target": any(mvtec_pass.values()) or bool(fallback.get("measured_quality_target_met")),
                   "selection_used_test_metrics": False,
                   "model_and_seed_selection_used_test_metrics": False,
                   "dataset_fallback_trigger_uses_frozen_mvtec_test_targets": True,
                   "limits": "Dataset point estimates only; failures and training-seed variability retained, no factory-reliability guarantee."}
        atomic_json(self.output / "study-results.json", summary)
        self.status.update(state="complete", finished_at=timestamp())
        self._save_status()
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="outputs/study-v2")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--execute", action="store_true", help="Run the declared study; otherwise print a dry-run plan")
    parser.add_argument("--mps-handoff", action="store_true", help="Acknowledge exclusive scheduler ownership granted by the root agent")
    parser.add_argument("--wait-existing", action="store_true", help="Wait for the original queue before any training")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(), indent=2))
        return
    if args.wait_existing:
        if not args.mps_handoff:
            raise RuntimeError("Waiting execution also requires explicit scheduler handoff authorization")
        root = args.root.resolve()
        waiting_path = root / args.output / "waiting-status.json"
        def update_waiting(queue):
            progress = None
            for item in queue.get("jobs", []):
                history_path = root / item.get("output", "") / "history.json"
                if item.get("state") == "running" and history_path.is_file():
                    history = json.loads(history_path.read_text())
                    progress = {"job": item["name"], "completed_epochs": len(history),
                                "latest_epoch_seconds": history[-1]["seconds"] if history else None}
            atomic_json(waiting_path, {"state": "waiting-for-initial-queue", "pid": os.getpid(),
                                      "queue": queue, "initial_queue_progress": progress,
                                      "updated_at_utc": timestamp()})
        print(json.dumps({"waiting_coordinator_pid": os.getpid(), "status": str(waiting_path)}), flush=True)
        try:
            completed = reconcile_queue(root / "outputs/protocol-v2-queue.json", wait=True, on_status=update_waiting)
            validate_completed_queue(completed)
        except BaseException as error:
            atomic_json(waiting_path, {"state": "failed", "pid": os.getpid(), "error": str(error),
                                      "updated_at_utc": timestamp()})
            raise
        atomic_json(waiting_path, {"state": "handoff-complete", "pid": os.getpid(), "updated_at_utc": timestamp()})
        # Load the final tested runner revision after waiting; edits to the
        # orchestration script cannot leave stale code alive for several hours.
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                 "--root", str(root), "--output", args.output,
                                 "--device", args.device, "--bootstrap-samples", str(args.bootstrap_samples),
                                 "--execute", "--mps-handoff"])
    StudyRunner(args.root, args.output, args.device, args.bootstrap_samples).run(args.mps_handoff, args.wait_existing)


if __name__ == "__main__":
    main()
