"""Exact pixel AP without a full pixel-level argsort; grouped image uncertainty."""
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from inspection.supervised import valid_anomaly_score

from .data import ValidationDataset
from .protocol import CONDITIONS, SELECTION_RULE


def harmonic(a, b):
    return 2 * a * b / (a + b) if a + b else 0.


def positive_threshold_ap(sorted_positive_arrays, sorted_negative_arrays):
    """Exact noninterpolated AP at positive score thresholds, including ties.

    Arrays must already be sorted ascending and finite. Negative-only thresholds
    add no recall, so their contribution is zero. No positives means AP=0 (the
    defined screen convention); all-positive data has AP=1. No full int64
    pixel-order permutation is allocated. Pooling concatenates POSITIVES only.
    """
    positives = [a for a in sorted_positive_arrays if len(a)]
    count = sum(len(a) for a in positives)
    if count == 0:
        return 0.
    values = positives[0] if len(positives) == 1 else np.concatenate(positives)
    thresholds, counts = np.unique(values, return_counts=True)
    true_positives = np.cumsum(counts[::-1], dtype=np.int64)[::-1]
    false_positives = np.zeros(len(thresholds), dtype=np.int64)
    for negative in sorted_negative_arrays:
        if len(negative):
            false_positives += len(negative) - np.searchsorted(negative, thresholds, side="left")
    precision = true_positives / (true_positives + false_positives)
    return float(np.sum(counts * precision, dtype=np.float64) / count)


class PixelSpools:
    """Append finite float32 scores; sort eight raw files in place on disk."""
    def __init__(self, directory, conditions=4):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.conditions = conditions
        self.paths = [[self.directory / f"condition-{i}-{kind}.f32" for kind in ("negative", "positive")]
                      for i in range(conditions)]
        self.handles = [[p.open("xb") for p in pair] for pair in self.paths]
        self.maps = []
        self.closed = False

    def append(self, scores, labels, condition):
        if self.closed or not 0 <= condition < self.conditions:
            raise ValueError("Invalid pixel spool state/condition")
        scores = np.asarray(scores, dtype=np.float32).ravel()
        labels = np.asarray(labels).ravel()
        if scores.shape != labels.shape or not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError("Pixel AP needs aligned finite scores and binary targets")
        positives = labels.astype(bool)
        scores[~positives].tofile(self.handles[condition][0])
        scores[positives].tofile(self.handles[condition][1])

    def metrics(self):
        for handles in self.handles:
            for handle in handles:
                handle.close()
        self.closed = True
        arrays = []
        for paths in self.paths:
            pair = []
            for path in paths:
                if not path.stat().st_size:
                    pair.append(np.empty(0, dtype=np.float32))
                else:
                    array = np.memmap(path, mode="r+", dtype=np.float32)
                    array.sort(kind="quicksort")  # in place, no pixel-order argsort
                    array.flush()
                    self.maps.append(array)
                    pair.append(array)
            arrays.append(pair)
        per_condition = [positive_threshold_ap([positive], [negative]) for negative, positive in arrays]
        pooled = positive_threshold_ap([p for _, p in arrays], [n for n, _ in arrays])
        counts = [{"negative": len(n), "positive": len(p)} for n, p in arrays]
        return pooled, per_condition, counts

    def close(self):
        for handles in self.handles:
            for handle in handles:
                if not handle.closed:
                    handle.close()
        for array in self.maps:
            array._mmap.close()
        self.maps.clear()
        self.closed = True


def image_metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape != (len(labels), len(CONDITIONS)) or not np.isfinite(scores).all():
        raise ValueError("Expected finite group-by-condition image scores")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Image AUROC requires both labels")
    return (float(roc_auc_score(np.repeat(labels, len(CONDITIONS)), scores.ravel())),
            [float(roc_auc_score(labels, scores[:, i])) for i in range(len(CONDITIONS))])


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
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)
            for batch in loader:
                _, logits = model(batch["image"].to(device))
                score = valid_anomaly_score(logits, batch["valid_mask"].to(device)).cpu().numpy()
                maps = torch.sigmoid(logits).cpu().numpy()
                valid = batch["valid_mask"].numpy().astype(bool)
                targets = batch["mask"].numpy()
                for i in range(len(score)):
                    group, condition = int(batch["group_index"][i]), int(batch["condition_index"][i])
                    label = int(batch["label"][i])
                    if image_labels[group] not in (-1, label) or np.isfinite(image_scores[group, condition]):
                        raise ValueError("Duplicate or inconsistent validation group")
                    image_scores[group, condition], image_labels[group] = float(score[i]), label
                    spool.append(maps[i][valid[i]], targets[i][valid[i]], condition)
            pooled_ap, condition_ap, counts = spool.metrics()
        finally:
            spool.close()
    pooled_auroc, condition_auroc = image_metrics(image_labels, image_scores)
    conditions = {name: {"image_auroc": condition_auroc[i], "pixel_ap": condition_ap[i],
                         "harmonic_mean": harmonic(condition_auroc[i], condition_ap[i]),
                         "pixel_counts": counts[i]} for i, name in enumerate(CONDITIONS)}
    return {"pooled": {"image_auroc": pooled_auroc, "pixel_ap": pooled_ap,
                       "harmonic_mean": harmonic(pooled_auroc, pooled_ap)},
            "conditions": conditions, "group_labels": image_labels.tolist(),
            "group_scores": image_scores.tolist(), "group_ids": dataset.bank["image_groups"],
            "bank_digest": dataset.bank["bank_digest"], "split_digest": dataset.bank["split_digest"],
            "source_images": groups, "variant_images": groups * len(CONDITIONS),
            "selection_rule": SELECTION_RULE,
            "pixel_ap_method": "exact noninterpolated AP at positive float32 score thresholds; ties include every score >= threshold",
            "no_positive_pixels_convention": "AP=0; no-positive condition cannot support localization success",
            "pixel_space": "model letterbox resolution with padding excluded; not native-resolution AP",
            "test_or_calibration_images_read": 0}


def grouped_paired_bootstrap(control, candidate, samples=1000, seed=4_000_042):
    if (control["group_ids"] != candidate["group_ids"] or control["group_labels"] != candidate["group_labels"]
            or control["bank_digest"] != candidate["bank_digest"]):
        raise ValueError("Paired bootstrap requires identical validation groups")
    labels = np.asarray(control["group_labels"])
    old, new = np.asarray(control["group_scores"]), np.asarray(candidate["group_scores"])
    original_old, _ = image_metrics(labels, old)
    original_new, _ = image_metrics(labels, new)
    indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        chosen = np.concatenate([generator.choice(group, len(group), replace=True) for group in indices])
        sampled_labels = np.repeat(labels[chosen], len(CONDITIONS))
        differences.append(float(roc_auc_score(sampled_labels, new[chosen].ravel())
                                 - roc_auc_score(sampled_labels, old[chosen].ravel())))
    lower, upper = np.quantile(differences, [.025, .975])
    return {"metric": "paired difference in pooled validation image AUROC (candidate minus control)",
            "difference": original_new - original_old, "percentile_95_interval": [float(lower), float(upper)],
            "bootstrap_samples": samples, "seed": seed, "original_image_groups": len(labels),
            "resampling": "stratified by original-image label, with all four variants retained together",
            "scope": "image AUROC only; no pixel AP interval; descriptive reused-validation uncertainty, not independent test evidence",
            "used_for_gate": False}
