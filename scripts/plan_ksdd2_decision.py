"""Record a decision-head hypothesis from saved validation scores; no inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root):
    source = root / "outputs/ksdd2-robustness/screen-result.json"
    screen = json.loads(source.read_text())
    conditions = ("clean", "brightness_0.8", "brightness_1.2", "jpeg_quality_60")
    records = {}
    for key in ("control", "candidate"):
        row = screen[key]
        labels = np.asarray(row["group_labels"], dtype=np.int64)
        scores = np.asarray(row["group_scores"], dtype=np.float64)
        if (scores.shape != (350, 4) or len(labels) != 350 or labels.sum() != 37
                or not np.isfinite(scores).all() or list(row["conditions"]) != list(conditions)
                or row["test_or_calibration_images_read"] != 0):
            raise ValueError("Expected complete saved validation-only grouped evidence")
        metrics = {
            name: {"image_auroc": float(roc_auc_score(labels, scores[:, i])),
                   "standardized_partial_auroc_fpr_0_to_0_1": float(roc_auc_score(labels, scores[:, i], max_fpr=.1)),
                   "pixel_ap": row["conditions"][name]["pixel_ap"]}
            for i, name in enumerate(conditions)
        }
        records[row["candidate"]] = {
            "selected_checkpoint_sha256": row["checkpoint_sha256"],
            "selected_epoch": row["best_epoch"],
            "conditions": metrics,
            "pooled_image_auroc": float(roc_auc_score(np.repeat(labels, 4), scores.flatten())),
            "pooled_standardized_partial_auroc_fpr_0_to_0_1": float(roc_auc_score(np.repeat(labels, 4), scores.flatten(), max_fpr=.1)),
            "pooled_pixel_ap": row["pooled"]["pixel_ap"],
            "normal_jpeg_scores_above_twice_clean": int(((scores[labels == 0, 3]) > 2 * scores[labels == 0, 0]).sum()),
            "normal_sources": int((labels == 0).sum()),
        }
    return {
        "schema_version": 1,
        "purpose": "Post-screen validation diagnostic motivating one bounded decision-head experiment; not an independent result or a score-selection sweep.",
        "source": str(source.relative_to(root)), "source_sha256": sha256(source),
        "generator_sha256": sha256(__file__),
        "split_digest": screen["control"]["split_digest"],
        "bank_digest": screen["control"]["bank_digest"],
        "original_validation_sources": 350, "normal_sources": 313, "defective_sources": 37,
        "conditions": list(conditions), "correlated_variants": 1400,
        "new_image_reads": 0, "calibration_reads": 0, "new_test_reads": 0,
        "model_selection_used_for_these_existing_scores": "Earlier pooled image-AUROC/pixel-AP harmonic mean; diagnostic partial AUROC does not reselect previous checkpoints.",
        "models": records,
        "hypothesis": "A detached decision head can learn image-level evidence beyond the fixed highest-1%-pixel average while preserving the segmentation objective.",
        "not_established": "No claim that these scores identify a specific causal false-alarm mechanism, that the head will pass q90 operating targets, or that reused validation/test sources are independent.",
        "bounded_next_action": "Exactly two fresh seed-42 trainings with matched acquisition augmentation: control top-1% scoring and detached image-decision head. Select by pooled low-FPR partial AUROC, guard each condition and pixel AP, freeze before clean-normal q90 calibration. No additional seeds or epochs beyond declared caps.",
        "stop_rule": "If this new head fails its frozen gate, stop this architecture search. New independent acquisition data is required for new independent reliability claims in every outcome.",
        "primary_references": [
            "https://arxiv.org/html/2104.06064",
            "https://github.com/vicoslab/mixed-segdec-net-comind2021/blob/master/models.py",
            "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
        ],
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/results/KSDD2_DECISION_PLANNING.json"
    payload = build(root)
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError("Refusing to replace different planning evidence")
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(path.relative_to(root)), "models": payload["models"]}, indent=2))
