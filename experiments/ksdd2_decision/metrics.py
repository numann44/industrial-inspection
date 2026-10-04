"""Score-aware low-FPR validation and exact disk-backed segmentation AP."""
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from inspection.preprocessing import prepare_image
from experiments.ksdd2_robustness.data import ValidationDataset
from experiments.ksdd2_robustness.metrics import PixelSpools
from .protocol import CONDITIONS, SELECTION_RULE


def image_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 2 or scores.shape != (len(labels), len(CONDITIONS))
            or not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}):
        raise ValueError("Expected finite group-by-condition scores with both image labels")

    def summarize(y, s):
        return {"image_pauc": float(roc_auc_score(y, s, max_fpr=.1)),
                "image_auroc": float(roc_auc_score(y, s)),
                "image_ap": float(average_precision_score(y, s))}

    return summarize(np.repeat(labels, len(CONDITIONS)), scores.ravel()), [summarize(labels, scores[:, index])
                                                                         for index in range(len(CONDITIONS))]


@torch.inference_mode()
def evaluate(model, bank_path, device, batch_size, scratch_root):
    dataset = ValidationDataset(bank_path)
    groups = dataset.bank["original_images"]
    image_scores = np.full((groups, len(CONDITIONS)), np.nan)
    image_labels = np.full(groups, -1)
    model.eval()
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pixel-ap-", dir=scratch_root) as temporary:
        spool = PixelSpools(temporary)
        try:
            for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False):
                logits, score = model(batch["image"].to(device), batch["valid_mask"].to(device))
                maps, scores = torch.sigmoid(logits).cpu().numpy(), score.cpu().numpy()
                valid, targets = batch["valid_mask"].numpy().astype(bool), batch["mask"].numpy()
                for index in range(len(scores)):
                    group, condition = int(batch["group_index"][index]), int(batch["condition_index"][index])
                    label = int(batch["label"][index])
                    if image_labels[group] not in (-1, label) or np.isfinite(image_scores[group, condition]):
                        raise ValueError("Duplicate or inconsistent validation group")
                    image_scores[group, condition], image_labels[group] = float(scores[index]), label
                    spool.append(maps[index][valid[index]], targets[index][valid[index]], condition)
            pooled_ap, condition_ap, counts = spool.metrics()
        finally:
            spool.close()
    pooled, per_condition = image_metrics(image_labels, image_scores)
    return {"pooled": {**pooled, "pixel_ap": pooled_ap},
            "conditions": {name: {**per_condition[index], "pixel_ap": condition_ap[index],
                                   "pixel_counts": counts[index]} for index, name in enumerate(CONDITIONS)},
            "group_labels": image_labels.tolist(), "group_scores": image_scores.tolist(),
            "group_ids": dataset.bank["image_groups"], "bank_digest": dataset.bank["bank_digest"],
            "split_digest": dataset.bank["split_digest"], "source_images": groups,
            "variant_images": groups * len(CONDITIONS), "selection_rule": SELECTION_RULE,
            "model_kind": model.model_kind, "score_definition": model.score_definition,
            "pauc_max_fpr": .1, "pauc_standardization": "McClish standardized partial ROC AUC via sklearn",
            "pixel_ap_method": "exact noninterpolated AP at positive float32 score thresholds; ties included",
            "no_positive_pixels_convention": "AP=0; no-positive condition cannot support localization success",
            "pixel_space": "model letterbox resolution with padding excluded; not native-resolution AP",
            "test_or_calibration_images_read": 0}


def grouped_paired_bootstrap(control, candidate, samples=1000, seed=5_000_042):
    if (control["group_ids"] != candidate["group_ids"] or control["group_labels"] != candidate["group_labels"]
            or control["bank_digest"] != candidate["bank_digest"] or samples < 1):
        raise ValueError("Paired bootstrap requires identical validation groups and positive sample count")
    labels = np.asarray(control["group_labels"])
    old, new = np.asarray(control["group_scores"]), np.asarray(candidate["group_scores"])
    original_old, _ = image_metrics(labels, old)
    original_new, _ = image_metrics(labels, new)
    indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    random = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        chosen = np.concatenate([random.choice(group, len(group), replace=True) for group in indices])
        y = np.repeat(labels[chosen], len(CONDITIONS))
        differences.append(float(roc_auc_score(y, new[chosen].ravel(), max_fpr=.1)
                                 - roc_auc_score(y, old[chosen].ravel(), max_fpr=.1)))
    lower, upper = np.quantile(differences, [.025, .975])
    return {"metric": "paired pooled standardized image partial AUROC difference, FPR<=0.1, candidate minus control",
            "difference": original_new["image_pauc"] - original_old["image_pauc"],
            "percentile_95_interval": [float(lower), float(upper)], "bootstrap_samples": samples,
            "seed": seed, "original_image_groups": len(labels),
            "resampling": "stratify original images by label; retain their four variants together",
            "scope": "descriptive reused-validation image uncertainty only; not independent test evidence or pixel AP CI",
            "used_for_gate": False}


@torch.inference_mode()
def calibrate(model, records, preprocess, batch_size, device, quantile=.9):
    if not records or any(row["label"] != 0 or row["original_split"] != "train" for row in records) or quantile != .9:
        raise ValueError("Only separate original clean normal calibration records and q90 are allowed")
    model.eval()
    scores = []
    # Read images only: no ground-truth mask is needed to choose a normal quantile.
    for start in range(0, len(records), batch_size):
        prepared = []
        for row in records[start:start + batch_size]:
            with Image.open(row["image"]) as image:
                prepared.append(prepare_image(image.convert("RGB"), preprocess))
        _, values = model(torch.stack([p[0] for p in prepared]).to(device),
                          torch.stack([p[1] for p in prepared]).to(device))
        scores.extend(values.cpu().tolist())
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite calibration scores")
    threshold = float(np.quantile(scores, .9, method="linear"))
    return {"split": "calibration", "normal_count": len(scores), "scores": scores,
            "quantile": .9, "quantile_method": "linear", "threshold": threshold,
            "decision_rule": ">=", "calibration_exceedances": sum(score >= threshold for score in scores),
            "model_kind": model.model_kind, "score_definition": model.score_definition,
            "warning": "An empirical clean-normal quantile does not guarantee future false-alarm rates."}
