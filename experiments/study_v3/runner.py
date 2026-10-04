"""Freeze or execute a three-run screen; never opens original test pixels."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from inspection.checkpointing import atomic_json, config_digest
from inspection.data import load_manifest
from scripts.run_study import scheduler_lock

from .bank import create_bank, load_bank
from .protocol import (PROTOCOL, SELECTION_RULE, config, continuation_manifest, gate,
                       immutable_json, sha256, source_hashes)
from .trainer import train


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def require_v2_complete(root):
    state = json.loads((root / "outputs/study-v2/status.json").read_text())
    results = root / "outputs/study-v2/study-results.json"
    if state.get("state") != "complete" or not results.is_file():
        raise ValueError("Finish the entire frozen v2 study before declaring this follow-up")
    return sha256(results)


def prepare(root, output, device="mps"):
    root, output = Path(root).resolve(), Path(output).resolve()
    if device not in ("cpu", "mps"):
        raise ValueError("Only local free CPU/MPS execution is supported")
    v2_results_hash = require_v2_complete(root)
    manifest_path, old_bank_path = root / "data/manifest.json", root / "data/validation-bank-v2/bank.json"
    manifest = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    with scheduler_lock(root / "outputs/mps-study.lock"):
        bank_path = output / "bank/bank.json"
        bank = load_bank(bank_path) if bank_path.exists() else create_bank(manifest_path, old_bank_path, output / "bank")
        if (bank["split_digest"] != manifest["split_digest"]
                or bank["old_bank_file_sha256"] != sha256(old_bank_path)):
            raise ValueError("Prepared bank no longer matches the original data/validation identity")
        declaration = {
            "protocol": PROTOCOL, "stage": "three-run-screen-only", "maximum_total_new_runs": 9,
            "screen_candidates": ["control", "A", "B"], "selection_rule": SELECTION_RULE,
            "checkpoint_ties": "keep earliest eligible epoch", "candidate_ties": "A before B",
            "gate": {"minimum_macro_h_gain": .01, "maximum_old_macro_h_regression": .02},
            "inputs": {"manifest": str(manifest_path), "manifest_file_sha256": sha256(manifest_path),
                       "split_digest": manifest["split_digest"], "bank": str(bank_path),
                       "bank_digest": bank["bank_digest"], "v2_results_sha256": v2_results_hash},
            "source_files": source_hashes(), "bank_family_counts": bank["family_counts"],
            "runs": [{"config": config(candidate, device),
                      "output": str(output / "runs" / f"metal_nut-{candidate}-seed42")}
                     for candidate in ("control", "A", "B")],
            "controlled_comparison": "Fresh control, A and B have identical training/selection budgets and the same v3 bank.",
            "historical_v2_control": {"checkpoint": "runs/protocol-v2-joint-256-seed42/checkpoint.pt",
                "sha256": sha256(root / "runs/protocol-v2-joint-256-seed42/checkpoint.pt"),
                "role": "preserved historical reference only; not the screening comparator"},
            "test_access": "This runner never opens test images or masks and performs no test evaluation.",
            "evidence_status": "All three previously evaluated MVTec categories are development-inspected/exploratory in v3.",
            "continuation": "At most six additional preregistered jobs; executor not yet implemented. See continuation.json after gate.",
            "uncertainty": "Validation is synthetic; neither passing the gate nor a previous test result proves new real-defect success.",
        }
        declaration["declaration_digest"] = config_digest(declaration)
        return immutable_json(output / "declaration.json", declaration)


def validate_declaration(declaration, root):
    payload = {k: v for k, v in declaration.items() if k != "declaration_digest"}
    if declaration.get("declaration_digest") != config_digest(payload):
        raise ValueError("Declaration digest mismatch")
    if declaration.get("protocol") != PROTOCOL or declaration["source_files"] != source_hashes():
        raise ValueError("Source or protocol changed after freezing the declaration")
    if declaration["inputs"]["v2_results_sha256"] != require_v2_complete(root):
        raise ValueError("V2 evidence changed after declaration")
    inputs = declaration["inputs"]
    if sha256(inputs["manifest"]) != inputs["manifest_file_sha256"]:
        raise ValueError("Original manifest changed")
    bank = load_bank(inputs["bank"])
    if bank["bank_digest"] != inputs["bank_digest"] or bank["split_digest"] != inputs["split_digest"]:
        raise ValueError("Frozen bank identity changed")
    if len(declaration["runs"]) != 3 or declaration["maximum_total_new_runs"] != 9:
        raise ValueError("Declared run budget differs")
    if (declaration["gate"] != {"minimum_macro_h_gain": .01, "maximum_old_macro_h_regression": .02}
            or declaration["selection_rule"] != SELECTION_RULE
            or declaration["screen_candidates"] != ["control", "A", "B"]):
        raise ValueError("Declared ranking or continuation gate differs")
    for row, candidate in zip(declaration["runs"], ("control", "A", "B")):
        if row["config"] != config(candidate, row["config"]["device"]):
            raise ValueError("Training settings differ from the declared bounded screen")
    return declaration


def execute(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    declaration = json.loads((output / "declaration.json").read_text())
    with scheduler_lock(root / "outputs/mps-study.lock"):
        validate_declaration(declaration, root)
        rows = []
        state = {"protocol": PROTOCOL, "state": "running", "pid": os.getpid(), "started_at": timestamp(),
                 "declaration_digest": declaration["declaration_digest"], "completed": []}
        try:
            for job in declaration["runs"]:
                run = Path(job["output"])
                state["current_candidate"] = job["config"]["candidate"]
                atomic_json(output / "status.json", state)
                checkpoint = train(declaration["inputs"]["manifest"], declaration["inputs"]["bank"],
                                   job["config"], run, declaration["declaration_digest"],
                                   resume=run / "last.pt" if (run / "last.pt").exists() else None,
                                   expected_sources=declaration["source_files"])
                row = {**checkpoint["best_validation"], "candidate": job["config"]["candidate"],
                       "checkpoint": str(run / "checkpoint.pt"), "checkpoint_sha256": sha256(run / "checkpoint.pt"),
                       "best_epoch": checkpoint["best_epoch"], "threshold": checkpoint["threshold"]}
                rows.append(row)
                state["completed"].append(row["candidate"])
            result = gate(rows[0], rows[1:], declaration["gate"]["minimum_macro_h_gain"],
                          declaration["gate"]["maximum_old_macro_h_regression"])
            immutable_json(output / "screen-result.json", result)
            continuation = continuation_manifest(result)
            immutable_json(output / "continuation.json", continuation)
            state.update(state="screen_complete", gate_state=result["state"], finished_at=timestamp(),
                         continuation_state=continuation["state"], current_candidate=None)
            atomic_json(output / "status.json", state)
            return result
        except BaseException as error:
            state.update(state="failed", error=f"{type(error).__name__}: {error}", failed_at=timestamp())
            atomic_json(output / "status.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("outputs/study-v3"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps", help="Declaration device; cannot change on resume")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Build immutable bank and freeze declaration; does not train")
    action.add_argument("--execute", action="store_true", help="Execute/resume only the already declared screen")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = prepare(root, output, args.device) if args.prepare else execute(root, output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
