"""Evaluate the passed two-run screen with fixed weights and exploratory labels."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from inspection.checkpointing import atomic_json, config_digest
from inspection.ksdd2 import load_manifest
from inspection.reporting import wilson_interval
from scripts.run_study import scheduler_lock
from experiments.ksdd2_robustness.data import load_bank
from experiments.ksdd2_robustness.protocol import check_identity, gate, immutable_json, sha256
from experiments.ksdd2_robustness.trainer import validate_checkpoint

STATUS = "exploratory"
CONDITIONS = ("original", "gaussian_blur_radius_1", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")


def now():
    return datetime.now(timezone.utc).isoformat()


def passed_inputs(root):
    """Read metadata/checkpoints only; no original-test or calibration pixels."""
    screen = root / "outputs/ksdd2-robustness"
    declaration = json.loads((screen / "declaration.json").read_text())
    check_identity(declaration)
    result = json.loads((screen / "screen-result.json").read_text())
    if result != gate(result["control"], result["candidate"]) or result["state"] != "awaiting_exploratory_evaluation":
        raise ValueError("A recorded passing two-run validation gate is required")
    status = json.loads((screen / "status.json").read_text())
    if not status.get("training_complete") or not status.get("calibration_performed"):
        raise ValueError("Training and gated calibration must finish before evaluation")
    inputs = declaration["inputs"]
    if sha256(inputs["manifest"]) != inputs["manifest_file_sha256"]:
        raise ValueError("Frozen manifest changed")
    manifest = load_manifest(inputs["manifest"])
    bank = load_bank(inputs["bank"], verify_sources=False)
    assets = json.loads((screen / "calibrated-assets.json").read_text())
    if [a["candidate"] for a in assets] != ["control", "acquisition_aug"]:
        raise ValueError("Exactly the two frozen calibrated candidates are required")
    for asset, job in zip(assets, declaration["runs"]):
        path = Path(asset["checkpoint"])
        if path != Path(job["output"]) / "checkpoint.pt" or sha256(path) != asset["checkpoint_sha256"]:
            raise ValueError("Frozen calibrated checkpoint identity differs")
        selected_path = path.with_name("selected.pt")
        selected = torch.load(selected_path, map_location="cpu", weights_only=True)
        validate_checkpoint(selected, job["config"], manifest, bank, declaration, torch.device(job["config"]["device"]), selected=True)
        row = result["control"] if asset["candidate"] == "control" else result["candidate"]
        c = torch.load(path, map_location="cpu", weights_only=True)
        identity = {"selected_checkpoint_sha256": sha256(selected_path), "gate_digest": config_digest(result)}
        if (row["checkpoint_sha256"] != identity["selected_checkpoint_sha256"] or asset["calibration_identity"] != identity
                or c.get("calibration_identity") != identity or c.get("checkpoint_kind") != "inference"
                or c.get("config_digest") != selected["config_digest"] or c.get("split_digest") != inputs["split_digest"]
                or c.get("threshold") != asset["threshold"] or not np.isfinite(asset["threshold"])):
            raise ValueError("Frozen selection/calibration identity differs")
        if (c["model_state"].keys() != selected["model_state"].keys()
                or any(not torch.equal(c["model_state"][key], value) for key, value in selected["model_state"].items())):
            raise ValueError("Calibration must not alter selected weights")
        calibration = c["calibration"]
        if (calibration["normal_count"] != 313 or len(calibration["scores"]) != 313 or calibration["quantile"] != .9
                or c["threshold"] != calibration["threshold"]
                or c["threshold"] != float(np.quantile(calibration["scores"], .9, method="linear"))):
            raise ValueError("Expected unchanged q90 evidence on 313 clean calibration normals")
    return declaration, result, assets, manifest


def validate_cached(value, asset, split_digest, kind, freeze_digest):
    if (value.get("checkpoint_sha256") != asset["checkpoint_sha256"] or value.get("threshold") != asset["threshold"]
            or value.get("split_digest") != split_digest or value.get("experimental_status") != STATUS
            or value.get("evaluation_freeze_digest") != freeze_digest):
        raise ValueError("Cached output has a different model, threshold, split or exploratory status")
    if kind == "evaluation" and value.get("pixel_metric_space") != "original":
        raise ValueError("Original-resolution localization is required")
    if kind == "stress" and tuple(value.get("results", {})) != CONDITIONS:
        raise ValueError("Stress condition set/order changed")
    return value


def validate_latency(value, asset, freeze_digest, benchmark_input):
    if (value.get("evaluation_freeze_digest") != freeze_digest or value.get("input") != benchmark_input
            or set(value.get("devices", {})) != {"cpu", "mps"}):
        raise ValueError("Cached latency provenance or device set differs")
    for device, row in value["devices"].items():
        if (row.get("device") != device or row.get("checkpoint_sha256") != asset["checkpoint_sha256"]
                or row.get("measurements") != 32 or row.get("warmup_forwards") != 3):
            raise ValueError("Cached latency method or model differs")
    return value


def rate_evidence(metrics):
    c = metrics["confusion"]
    return {"defect_recall_interval": wilson_interval(c["true_positive"], c["true_positive"] + c["false_negative"]),
            "normal_false_alarm_rate_interval": wilson_interval(c["false_positive"], c["false_positive"] + c["true_negative"]),
            "precision_interval": wilson_interval(c["true_positive"], c["true_positive"] + c["false_positive"]),
            "method": "95% Wilson image-level intervals; exploratory reused data, no domain-shift guarantee"}


def target_met(metrics):
    return bool(metrics["defect_recall"] >= .9 and metrics["normal_false_alarm_rate"] <= .1)


def execute(root):
    root = root.resolve()
    out = root / "outputs/ksdd2-robustness-evaluation"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with scheduler_lock(root / "outputs/mps-study.lock"):
        declaration, gate_result, assets, manifest = passed_inputs(root)
        wrapper = Path(__file__).resolve()
        benchmark_record = next(r for r in manifest["splits"]["calibration"] if r["label"] == 0)
        benchmark_input = {"source": benchmark_record["relative_image"], "image_sha256": benchmark_record["image_sha256"]}
        freeze = {"protocol": "ksdd2-robustness-exploratory-evaluation-1", "experimental_status": STATUS,
                  "training_declaration_digest": declaration["declaration_digest"], "gate_digest": config_digest(gate_result),
                  "assets": assets, "manifest_file_sha256": declaration["inputs"]["manifest_file_sha256"],
                  "split_digest": manifest["split_digest"], "evaluation_wrapper_sha256": sha256(wrapper),
                  "native_stream_evaluator_sha256": sha256(root / "scripts/ksdd2_native_stream.py"),
                  "frozen_sources": declaration["source_files"], "device": "mps", "batch_size": 8,
                  "pixel_space": "original", "bootstrap_samples": 1000, "conditions": list(CONDITIONS),
                  "benchmark": {"devices": ["cpu", "mps"], "measurements": 32, "warmup": 3,
                                "input": benchmark_input},
                  "selection": "acquisition_aug selected by prior validation gate; evaluation never selects or tunes a model",
                  "scope": "Reused KSDD2 original test and its paired perturbations; development-inspected/exploratory, not independent."}
        immutable_json(out / "pre-evaluation-freeze.json", freeze)
        freeze_digest = config_digest(freeze)
        state = {"state": "running", "pid": os.getpid(), "started_at_utc": now(), "completed": [],
                 "freeze_digest": config_digest(freeze), "experimental_status": STATUS}
        atomic_json(out / "status.json", state)
        try:
            from scripts.ksdd2_native_stream import evaluate
            from inspection.evaluate import stress_test_checkpoint, benchmark_checkpoint
            results = []
            for asset in assets:
                name = asset["candidate"]
                folder = out / name
                folder.mkdir(exist_ok=True)
                state["current"] = name + ": original/native evaluation"
                atomic_json(out / "status.json", state)
                path = folder / "evaluation.json"
                if path.exists():
                    evaluation = validate_cached(json.loads(path.read_text()), asset, manifest["split_digest"], "evaluation", freeze_digest)
                else:
                    evaluation = evaluate(declaration["inputs"]["manifest"], asset["checkpoint"], device="mps", batch_size=8,
                                          bootstrap_samples=1000, scratch_root=folder / "scratch")
                    evaluation["evaluation_freeze_digest"] = freeze_digest
                    evaluation["validation_selection"] = declaration["selection_rule"]
                    evaluation["measured_quality_target_met"] = target_met(evaluation)
                    evaluation["quality_target_scope"] = "exploratory reused-test point estimates, not independent evidence"
                    validate_cached(evaluation, asset, manifest["split_digest"], "evaluation", freeze_digest)
                    atomic_json(path, evaluation)
                print(json.dumps({"candidate": name, "phase": "original_complete", "confusion": evaluation["confusion"],
                                  "pixel_ap": evaluation["pixel_average_precision"]}), flush=True)
                state["current"] = name + ": paired stress"
                atomic_json(out / "status.json", state)
                path = folder / "stress.json"
                if path.exists():
                    stress = validate_cached(json.loads(path.read_text()), asset, manifest["split_digest"], "stress", freeze_digest)
                else:
                    stress = stress_test_checkpoint(Path(declaration["inputs"]["manifest"]), Path(asset["checkpoint"]), "mps", 8)
                    stress["experimental_status"] = STATUS
                    stress["evaluation_freeze_digest"] = freeze_digest
                    for metrics in stress["results"].values():
                        metrics["uncertainty"] = rate_evidence(metrics)
                        metrics["measured_quality_target_met"] = target_met(metrics)
                    validate_cached(stress, asset, manifest["split_digest"], "stress", freeze_digest)
                    atomic_json(path, stress)
                if stress["results"]["original"]["confusion"] != evaluation["confusion"]:
                    raise ValueError("Repeated original inference decisions differ between native and stress evaluation")
                state["current"] = name + ": CPU/MPS timing"
                atomic_json(out / "status.json", state)
                path = folder / "latency.json"
                if path.exists():
                    latency = validate_latency(json.loads(path.read_text()), asset, freeze_digest, benchmark_input)
                else:
                    image = Path(benchmark_record["image"])
                    if sha256(image) != benchmark_input["image_sha256"]:
                        raise ValueError("Frozen benchmark input changed")
                    latency = {"evaluation_freeze_digest": freeze_digest, "input": benchmark_input,
                               "devices": {device: benchmark_checkpoint(Path(asset["checkpoint"]), image, device, 32, 3)
                                           for device in ("cpu", "mps")}}
                    validate_latency(latency, asset, freeze_digest, benchmark_input)
                    immutable_json(path, latency)
                results.append({"candidate": name, "evaluation": str(folder / "evaluation.json"),
                                "stress": str(folder / "stress.json"), "latency": str(path),
                                "checkpoint_sha256": asset["checkpoint_sha256"]})
                state["completed"].append(name)
                atomic_json(out / "status.json", state)
            immutable_json(out / "results.json", {"freeze_digest": config_digest(freeze), "experimental_status": STATUS,
                           "results": results, "selected_before_evaluation": "acquisition_aug", "thresholds_changed": False})
            state.update(state="complete", current=None, finished_at_utc=now())
            atomic_json(out / "status.json", state)
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", failed_at_utc=now())
            atomic_json(out / "status.json", state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    execute(parser.parse_args().root)
