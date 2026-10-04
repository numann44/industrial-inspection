"""Immutable identity and validation-only decisions for exactly two runs."""
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from inspection.checkpointing import atomic_json, config_digest

PROTOCOL = "ksdd2-robustness-screen-1"
CONDITIONS = ("clean", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")
SELECTION_RULE = "H(pooled cross-condition image AUROC, exact pooled model-space pixel AP); padding excluded"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [p for p in (root / "experiments/ksdd2_robustness").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths += list((root / "src/inspection").glob("*.py"))
    paths += [root / "scripts/run_study.py"]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "numpy", "Pillow", "scikit-learn", "scipy")},
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Immutable record differs: {path}")
    else:
        atomic_json(path, value)
    return value


def config(candidate, device="mps"):
    return {"protocol": PROTOCOL, "candidate": candidate, "base_channels": 16, "training_seed": 42,
            "batch_size": 8, "epochs": 100, "patience_epochs": 15, "selection_every": 5,
            "learning_rate": .0003, "positive_pixel_weight": 3., "threshold_quantile": .9,
            "preprocess": {"mode": "letterbox", "height": 640, "width": 256},
            "device": device, "cpu_threads": 1,
            "augmentation": {"unchanged_probability": .5, "brightness_range": [.75, 1.25],
                             "jpeg_quality_inclusive": [60, 95], "order": ["brightness", "jpeg", "letterbox"]}}


def gate(control, candidate):
    if control["candidate"] != "control" or candidate["candidate"] != "acquisition_aug":
        raise ValueError("Exactly the fresh control and acquisition_aug must be compared")
    if any(control[key] != candidate[key] for key in ("bank_digest", "split_digest")):
        raise ValueError("Control/candidate bank or split differs")
    for row in (control, candidate):
        values = [row["pooled"]["harmonic_mean"], *[row["conditions"][c]["harmonic_mean"] for c in CONDITIONS]]
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid validation metrics")
    gain = candidate["pooled"]["harmonic_mean"] - control["pooled"]["harmonic_mean"]
    changes = {c: candidate["conditions"][c]["harmonic_mean"] - control["conditions"][c]["harmonic_mean"]
               for c in CONDITIONS}
    passed = gain >= .01 and all(change >= -.01 for change in changes.values())
    return {"protocol": PROTOCOL, "state": "awaiting_exploratory_evaluation" if passed else "closed_failed_gate",
            "selected": candidate if passed else None, "control": control, "candidate": candidate,
            "pooled_h_gain": gain, "condition_h_changes": changes,
            "rule": {"minimum_pooled_h_gain": .01, "maximum_each_condition_h_regression": .01},
            "test_or_calibration_metrics_used_for_selection": False, "further_training_allowed_by_this_screen": False,
            "independence": "Any reused KSDD2 test is development-inspected/exploratory for this method; no new independent claim."}


def check_identity(declaration):
    if declaration["source_files"] != source_hashes():
        raise ValueError("Source changed after declaration")
    if declaration["environment"] != environment():
        raise ValueError("Environment changed after declaration")
    body = {k: v for k, v in declaration.items() if k != "declaration_digest"}
    if declaration.get("declaration_digest") != config_digest(body):
        raise ValueError("Declaration digest mismatch")
