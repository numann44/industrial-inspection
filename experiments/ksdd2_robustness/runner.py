"""Freeze and execute exactly two robustness runs, then stop at the declared gate."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from inspection.checkpointing import atomic_json, config_digest
from inspection.ksdd2 import load_manifest
from scripts.run_study import scheduler_lock

from .data import create_bank, load_bank, verify_bank_membership
from .metrics import grouped_paired_bootstrap
from .protocol import (PROTOCOL, CONDITIONS, SELECTION_RULE, check_identity, config, environment,
                       gate, immutable_json, sha256, source_hashes)
from .trainer import train, calibrate_after_gate


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def validate_output(root, output):
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("Operational outputs must stay in the project's ignored outputs tree, outside frozen sources")


def prior_evidence(root):
    v2_status = json.loads((root / "outputs/study-v2/status.json").read_text())
    v3_gate_path = root / "outputs/study-v3/screen-result.json"
    v3_gate = json.loads(v3_gate_path.read_text())
    if v2_status.get("state") != "complete" or v3_gate.get("state") != "stopped_no_validation_gain":
        raise ValueError("This bounded follow-up requires completed v2 and the closed failed v3 gate")
    return {"v2_results_sha256": sha256(root / "outputs/study-v2/study-results.json"),
            "v3_failed_gate_sha256": sha256(v3_gate_path),
            "validation_diagnostic_sha256": sha256(root / "outputs/ksdd2-validation-stress-planning.json")}


def prepare(root, output, device="mps"):
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_output(root, output)
    if device not in ("cpu", "mps"):
        raise ValueError("Only local CPU/MPS execution is allowed")
    evidence = prior_evidence(root)
    manifest_path = root / "data/ksdd2-manifest.json"
    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    with scheduler_lock(root / "outputs/mps-study.lock"):
        bank_path = output / "validation-bank.json"
        bank = load_bank(bank_path) if bank_path.exists() else create_bank(manifest_path, bank_path)
        verify_bank_membership(bank, manifest)
        if bank["original_images"] != 350 or bank["variant_images"] != 1400:
            raise ValueError("The production screen requires the original 350 validation records and four variants")
        declaration = {"protocol": PROTOCOL, "maximum_new_training_runs": 2,
            "conditions": list(CONDITIONS), "selection_rule": SELECTION_RULE,
            "checkpoint_ties": "earliest eligible epoch", "gate": {"minimum_pooled_h_gain": .01,
                "maximum_clean_h_regression": .01, "maximum_each_condition_h_regression": .01},
            "selection_schedule": "every 5 epochs and final epoch; 15 stale training epochs equals three failed checks",
            "calibration_order": "only after both runs finish, both pooled-selected weights are frozen and the global gate passes",
            "source_files": source_hashes(), "environment": environment(), "prior_evidence": evidence,
            "inputs": {"manifest": str(manifest_path), "manifest_file_sha256": sha256(manifest_path),
                       "split_digest": manifest["split_digest"], "bank": str(bank_path), "bank_digest": bank["bank_digest"]},
            "runs": [{"config": config(candidate, device), "output": str(output / "runs" / f"{candidate}-seed42")}
                     for candidate in ("control", "acquisition_aug")],
            "bootstrap": {"samples": 1000, "seed": 4_000_042, "metric": "paired pooled image AUROC only",
                          "group": "original validation image, retaining all four variants", "used_for_gate": False},
            "test_access": "This screen never reads original test images or masks and performs no test evaluation.",
            "independence": "All reused KSDD2 tests are development-inspected/exploratory for this method; no new independent holdout exists.",
            "stop_boundary": "Exactly two fresh runs. Failed gate closes this experiment; passed gate permits calibration then awaits exploratory evaluation. No further training.",
            "hypothesis": "A balanced unchanged/acquisition-augmented training policy may reduce camera-artifact sensitivity without sacrificing clean defect localization."}
        declaration["declaration_digest"] = config_digest(declaration)
        return immutable_json(output / "declaration.json", declaration)


def validate_declaration(declaration, root, output):
    validate_output(root, output)
    check_identity(declaration)
    if (declaration["protocol"] != PROTOCOL or declaration["maximum_new_training_runs"] != 2
            or declaration["conditions"] != list(CONDITIONS) or declaration["selection_rule"] != SELECTION_RULE
            or declaration["gate"] != {"minimum_pooled_h_gain": .01, "maximum_clean_h_regression": .01,
                                       "maximum_each_condition_h_regression": .01}):
        raise ValueError("Declared budget/ranking/gate differs")
    if declaration["prior_evidence"] != prior_evidence(root):
        raise ValueError("Prior evidence identity differs")
    inputs = declaration["inputs"]
    if sha256(inputs["manifest"]) != inputs["manifest_file_sha256"]:
        raise ValueError("Original manifest changed")
    manifest, bank = load_manifest(inputs["manifest"]), load_bank(inputs["bank"])
    verify_bank_membership(bank, manifest)
    if bank["bank_digest"] != inputs["bank_digest"] or manifest["split_digest"] != inputs["split_digest"]:
        raise ValueError("Bank or split differs")
    if len(declaration["runs"]) != 2:
        raise ValueError("Exactly two runs are allowed")
    for job, candidate in zip(declaration["runs"], ("control", "acquisition_aug")):
        if (job["config"] != config(candidate, job["config"]["device"])
                or Path(job["output"]).resolve() != output / "runs" / f"{candidate}-seed42"):
            raise ValueError("Run settings or output path differ from the bounded screen")
    return declaration


def execute(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_output(root, output)
    declaration = json.loads((output / "declaration.json").read_text())
    with scheduler_lock(root / "outputs/mps-study.lock"):
        validate_declaration(declaration, root, output)
        state = {"protocol": PROTOCOL, "state": "running", "pid": os.getpid(), "started_at": timestamp(),
                 "declaration_digest": declaration["declaration_digest"], "completed": []}
        rows = []
        try:
            for job in declaration["runs"]:
                output_run = Path(job["output"])
                state["current_candidate"] = job["config"]["candidate"]
                atomic_json(output / "status.json", state)
                selected = train(declaration["inputs"]["manifest"], declaration["inputs"]["bank"], job["config"],
                                 output_run, declaration, resume=output_run / "last.pt" if (output_run / "last.pt").exists() else None)
                rows.append({**selected["best_validation"], "candidate": job["config"]["candidate"],
                             "checkpoint": str(output_run / "selected.pt"), "checkpoint_sha256": sha256(output_run / "selected.pt"),
                             "best_epoch": selected["best_epoch"], "completed_epochs": selected["completed_epochs"]})
                state["completed"].append(job["config"]["candidate"])
            result = gate(rows[0], rows[1])
            immutable_json(output / "screen-result.json", result)
            # Bootstrap is descriptive only and cannot alter the recorded gate.
            uncertainty = grouped_paired_bootstrap(rows[0], rows[1], declaration["bootstrap"]["samples"], declaration["bootstrap"]["seed"])
            immutable_json(output / "paired-validation-uncertainty.json", uncertainty)
            if result["state"] == "awaiting_exploratory_evaluation":
                assets = []
                for job in declaration["runs"]:
                    checkpoint = calibrate_after_gate(declaration["inputs"]["manifest"], declaration["inputs"]["bank"],
                                                      job["config"], job["output"], declaration, result)
                    path = Path(job["output"]) / "checkpoint.pt"
                    assets.append({"candidate": job["config"]["candidate"], "checkpoint": str(path),
                                   "checkpoint_sha256": sha256(path), "threshold": checkpoint["threshold"],
                                   "calibration_identity": checkpoint["calibration_identity"]})
                immutable_json(output / "calibrated-assets.json", assets)
            state.update(state=result["state"], current_candidate=None, finished_at=timestamp(),
                         training_complete=True, further_training_allowed=False,
                         calibration_performed=result["state"] == "awaiting_exploratory_evaluation")
            atomic_json(output / "status.json", state)
            return result
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", failed_at=timestamp())
            atomic_json(output / "status.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("outputs/ksdd2-robustness"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = prepare(root, output, args.device) if args.prepare else execute(root, output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("control", "candidate", "selected")}, indent=2))


if __name__ == "__main__":
    main()
