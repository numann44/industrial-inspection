"""Immutable declarations, provenance and a validation-only continuation gate."""
import hashlib
import json
from pathlib import Path

from inspection.checkpointing import atomic_json, config_digest

PROTOCOL = "study-v3-screen-1"
SELECTION_RULE = "arithmetic mean of per-family H(image AUROC, pixel AP), each with the same clean validation controls"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [p for p in (root / "experiments/study_v3").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths += list((root / "src/inspection").glob("*.py"))
    paths += [root / "scripts/run_study.py"]  # shared scheduler lock
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Immutable record differs: {path}")
    else:
        atomic_json(path, value)
    return value


def config(candidate, device="mps", seed=42):
    return {"protocol": PROTOCOL, "category": "metal_nut", "candidate": candidate,
            "model_kind": "joint_reconstruction_segmentation", "image_size": 256,
            "base_channels": 16, "batch_size": 8, "epochs": 100,
            "learning_rate": .0003, "normal_probability": .25,
            "training_seed": seed, "data_seed": 42, "threshold_quantile": .9,
            "selection_every": 5, "cpu_threads": 1, "device": device}


def gate(control, candidates, minimum_gain=.01, maximum_old_regression=.02):
    """Test metrics never enter this function. Equal scores prefer A by ID."""
    if {c["candidate"] for c in candidates} != {"A", "B"} or len(candidates) != 2:
        raise ValueError("The screen must contain exactly candidates A and B")
    for row in [control, *candidates]:
        if row["bank_digest"] != control["bank_digest"] or row["split_digest"] != control["split_digest"]:
            raise ValueError("Candidate/control bank or split differs")
        for name in ("macro_h", "old_macro_h"):
            if not 0 <= row[name] <= 1:
                raise ValueError("Invalid validation score")
    eligible = [c for c in candidates if c["macro_h"] >= control["macro_h"] + minimum_gain
                and c["old_macro_h"] >= control["old_macro_h"] - maximum_old_regression]
    winner = sorted(eligible, key=lambda c: (-c["macro_h"], c["candidate"]))[0] if eligible else None
    return {"state": "passed" if winner else "stopped_no_validation_gain", "selected": winner,
            "control": control, "candidates": candidates, "minimum_macro_h_gain": minimum_gain,
            "maximum_old_family_macro_h_regression": maximum_old_regression,
            "test_metrics_used_for_selection": False,
            "warning": "Synthetic validation is not evidence of real-defect target success."}


def continuation_manifest(result):
    jobs = []
    if result["selected"]:
        winner = result["selected"]["candidate"]
        jobs = [{"category": "metal_nut", "seed": seed, "candidate": winner} for seed in (43, 44)]
        jobs += [{"category": category, "seed": 42, "candidate": winner}
                 for category in ("screw", "transistor", "cable")]
        jobs += [{"category": "cable", "seed": 42, "candidate": "unchanged_v2_control"}]
    return {"protocol": PROTOCOL, "screen_result_digest": config_digest(result),
            "maximum_new_runs_including_screen": 9, "screen_runs": 3,
            "continuation_jobs": jobs, "executor_implemented": False,
            "state": "awaiting_continuation_implementation" if jobs else "stopped",
            "required_before_execution": ["Implement and review the continuation executor",
                "Keep existing audited category splits; no repartitioning",
                "Freeze both cable weights, seed choices and q90 thresholds before its first test read",
                "Verify cable test images/predictions were not previously inspected for method decisions"],
            "evidence_status": {"metal_nut": "development-inspected/exploratory",
                "screw": "development-inspected/exploratory", "transistor": "development-inspected/exploratory",
                "cable": "prospective method/category comparison only; separate category weights"},
            "same_category_independent_holdout": "No new same-category holdout is currently available."}
