"""Fixed two-run budget, source identity and threshold-free validation gate."""
from pathlib import Path

from inspection.checkpointing import config_digest
from experiments.ksdd2_robustness.protocol import CONDITIONS, environment, immutable_json, sha256

PROTOCOL = "ksdd2-detached-decision-screen-1"
CANDIDATES = ("control", "decision_head")
SELECTION_RULE = "pooled standardized image partial AUROC over FPR [0,0.1]; all four conditions; earliest exact tie"
GATE_RULE = {"minimum_positive_pauc_gain": True, "minimum_pauc_shortfall_reduction": .1,
             "maximum_each_condition_pauc_regression": .01, "maximum_pooled_pixel_ap_regression": .01,
             "maximum_each_condition_pixel_ap_regression": .01}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for directory in ("experiments/ksdd2_decision", "experiments/ksdd2_robustness"):
        paths += [p for p in (root / directory).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    paths += list((root / "src/inspection").glob("*.py")) + [root / "scripts/run_study.py"]
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def config(candidate, device="mps"):
    if candidate not in CANDIDATES or device not in ("cpu", "mps"):
        raise ValueError("Unknown local decision candidate/device")
    from experiments.ksdd2_robustness.protocol import config as prior_config
    return {**prior_config("acquisition_aug", device), "protocol": PROTOCOL, "candidate": candidate,
            "head_seed": 9_000_042, "classification_loss_weight": 1., "max_fpr": .1,
            "classification_gradients_to_segmentation": False}


def gate(control, candidate):
    if (control["candidate"], candidate["candidate"]) != CANDIDATES:
        raise ValueError("Exactly the fresh control and decision_head must be compared")
    if any(control[key] != candidate[key] for key in ("bank_digest", "split_digest")):
        raise ValueError("Control/candidate bank or split differs")
    for row in (control, candidate):
        for summary in (row["pooled"], *[row["conditions"][name] for name in CONDITIONS]):
            if any(not 0 <= summary[name] <= 1 for name in ("image_pauc", "pixel_ap")):
                raise ValueError("Invalid finite validation metrics")
    gain = candidate["pooled"]["image_pauc"] - control["pooled"]["image_pauc"]
    shortfall = 1 - control["pooled"]["image_pauc"]
    condition_changes = {name: {metric: candidate["conditions"][name][metric] - control["conditions"][name][metric]
                                for metric in ("image_pauc", "pixel_ap")} for name in CONDITIONS}
    pixel_change = candidate["pooled"]["pixel_ap"] - control["pooled"]["pixel_ap"]
    passed = (gain > 0 and gain >= .1 * shortfall and pixel_change >= -.01
              and all(change[metric] >= -.01 for change in condition_changes.values()
                      for metric in ("image_pauc", "pixel_ap")))
    return {"protocol": PROTOCOL, "state": "awaiting_exploratory_evaluation" if passed else "closed_failed_gate",
            "selected": candidate if passed else None, "control": control, "candidate": candidate,
            "pooled_pauc_gain": gain, "control_pauc_shortfall": shortfall,
            "pauc_shortfall_reduction": gain / shortfall if shortfall > 0 else None,
            "pooled_pixel_ap_change": pixel_change, "condition_changes": condition_changes, "rule": GATE_RULE,
            "test_or_calibration_metrics_used_for_selection": False, "further_training_allowed_by_this_screen": False,
            "independence": "All reused KSDD2 tests are exploratory; no new independent holdout exists."}


def check_identity(declaration):
    if declaration["source_files"] != source_hashes():
        raise ValueError("Source changed after declaration")
    if declaration["environment"] != environment():
        raise ValueError("Environment changed after declaration")
    if declaration.get("declaration_digest") != config_digest({k: v for k, v in declaration.items() if k != "declaration_digest"}):
        raise ValueError("Declaration digest mismatch")
