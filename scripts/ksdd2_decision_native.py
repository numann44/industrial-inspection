"""Memory-bounded native-resolution evaluation of both decision-screen scores.

This wrapper is intentionally exploratory: previous KSDD2 test failures informed
the robustness and learned-decision interventions. It neither selects weights nor changes thresholds.
Only a batch of model-resolution maps and one native map/mask are retained;
float32 pixel scores are sorted in temporary disk-backed spools for exact AP.
The caller owns serialization of GPU work and separate latency/gallery work.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from experiments.ksdd2_robustness.metrics import PixelSpools
from scripts.ksdd2_decision_inference import evaluation_context, additional_sources, _positive_integer
from inspection.evaluate import compute_metrics
from inspection.preprocessing import decode_image, prepare_image, restore_map
from inspection.reporting import checkpoint_provenance, defect_area_groups, image_uncertainty


def _append_native(spool, record, model_map, transform):
    """Release each original mask and restored prediction after appending scores."""
    native = restore_map(model_map, transform)
    with Image.open(record["mask"]) as image:
        mask = np.asarray(image)
        if (mask.ndim != 2 or mask.shape != native.shape
                or not np.isin(mask, (0, 255)).all()):
            raise ValueError("Native prediction and original binary mask must align")
        target = mask > 0
    width, height = record["original_size"]
    if native.shape != (height, width):
        raise ValueError("Native geometry differs from the audited image dimensions")
    spool.append(native, target, condition=0)


@torch.inference_mode()
def evaluate(manifest_path, checkpoint_path, device="mps", batch_size=8,
             bootstrap_samples=1000, scratch_root=None):
    """Return the stock evaluation schema, excluding gallery and latency fields.

    ``scratch_root`` optionally chooses a volume for float32 pixel score files;
    the unique temporary child and all score files are removed on success/error.
    Normal-only calibration data and optimization splits are never opened.
    """
    _positive_integer(batch_size, "batch_size")
    _positive_integer(bootstrap_samples, "bootstrap_samples")
    manifest_path, checkpoint_path = Path(manifest_path), Path(checkpoint_path)
    manifest, engine = evaluation_context(manifest_path, checkpoint_path, device, batch_size)
    checkpoint = engine.checkpoint
    records = manifest["splits"]["test"]
    if scratch_root is not None:
        scratch_root = Path(scratch_root)
        scratch_root.mkdir(parents=True, exist_ok=True)
    scores = []
    with tempfile.TemporaryDirectory(prefix="ksdd2-native-ap-", dir=scratch_root) as temporary:
        spool = PixelSpools(temporary, conditions=1)
        try:
            for start in range(0, len(records), batch_size):
                batch_records = records[start:start + batch_size]
                prepared = []
                for record in batch_records:
                    image = decode_image(record["image"])
                    if list(image.size) != record["original_size"]:
                        raise ValueError("Decoded image differs from the audited native dimensions")
                    prepared.append(prepare_image(image, engine.preprocess))
                images = torch.stack([item[0] for item in prepared])
                valid = torch.stack([item[1] for item in prepared])
                maps, values = engine.predict_tensor(images, valid)
                batch_maps = maps.cpu().numpy()
                scores.extend(values.cpu().tolist())
                for index, record in enumerate(batch_records):
                    _append_native(spool, record, batch_maps[index, 0], prepared[index][2])
                del images, valid, maps, values, batch_maps, prepared
            pooled_ap, _, counts = spool.metrics()
        finally:
            spool.close()
    pixel_counts = counts[0]
    positive_pixels = pixel_counts["positive"]
    pixels = pixel_counts["negative"] + positive_pixels
    # Match the reporting schema's undefined-localization convention rather than
    # the validation screen's deliberate no-positive AP=0 selection convention.
    native_ap = pooled_ap if positive_pixels else None
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    empty = np.zeros((len(records), 1, 1, 1), dtype=np.float32)
    metrics = compute_metrics(labels, scores, empty, empty, engine.threshold, include_pixel_metric=False)
    groups = defaultdict(list)
    for record, score in zip(records, scores):
        if record["label"]:
            groups[f"{record['category']}/{record['defect']}"].append(score >= engine.threshold)
    metrics.update({
        "pixel_average_precision": native_ap,
        "pixel_average_precision_by_category": {engine.category: {
            "average_precision": native_ap, "pixels": pixels,
            "positive_pixels": positive_pixels, "images": len(records)}},
        "pixel_metric_space": "original",
        "pixel_aggregation": "macro mean of per-category pixel average precision",
        "pixel_resize": "crop letterbox padding, bilinear restore predictions, unchanged original masks",
        "pixel_ap_method": "exact noninterpolated AP at positive float32 score thresholds; ties include every score >= threshold",
        "no_positive_pixels_convention": "AP=null; no positive masks cannot support localization success",
        "evaluation_resolution": "original resolution per image",
        "recall_by_defect": {name: {"recall": float(np.mean(values)), "images": len(values),
                                     "detected": int(sum(values))} for name, values in sorted(groups.items())},
        "manifest_digest": manifest["digest"], "split_digest": manifest["split_digest"],
        "checkpoint": str(checkpoint_path.resolve()),
        "model_sha256": engine.model_sha256, "model_kind": engine.model_kind,
        "candidate": engine.candidate, "declaration_digest": checkpoint["declaration_digest"],
        "calibration_identity": checkpoint["calibration_identity"],
        "preprocess": engine.preprocess, "experimental_status": "exploratory",
        "dataset": manifest["dataset"],
        "threshold_source": checkpoint.get("threshold_source", "checkpoint normal calibration split"),
        "score_definition": engine.score_definition,
        "map_definition": "sigmoid segmentation pixel activations; independent of image score units",
        "uncertainty": image_uncertainty(labels, scores, engine.threshold, bootstrap_samples=bootstrap_samples),
        "defect_area_groups": defect_area_groups(records, scores, engine.threshold),
        "class_prevalence": {"normal": int(np.sum(labels == 0)), "defective": int(np.sum(labels == 1)),
                             "defective_fraction": float(np.mean(labels))},
        "predictions": [{"image": record["image"], "category": record["category"],
                         "defect": record["defect"], "label": record["label"],
                         "score": float(score), "predicted_defective": bool(score >= engine.threshold)}
                        for record, score in zip(records, scores)],
        "training_supervision": "real defective and normal training images with pixel annotations",
        "comparison_scope": "separate supervised task; not directly comparable to normal-only MVTec",
        "validation_selection": checkpoint.get("model_selection"),
        "validation_best_epoch": checkpoint.get("best_epoch"),
        "validation_best_metric": checkpoint.get("best_metric"),
        "training_seed": checkpoint.get("config", {}).get("training_seed"),
        "calibration_normal_count": checkpoint.get("calibration", {}).get("normal_count"),
        "product_group_metadata": "not supplied; image-similarity checks do not establish batch/product independence",
        "limitations": [
            "Exploratory reused-test evidence: previous KSDD2 test failures informed the robustness and learned-decision interventions.",
            "Real-defect supervision does not establish recognition of previously unseen defect families.",
            "Restored maps cannot recover visual details lost during input resampling.",
            "Calibration quantiles do not guarantee future false-alarm rates.",
            "Provided image identities do not establish independent physical products or acquisition batches.",
            "Classification gradients are detached: localization differences can follow checkpoint selection, not direct backbone improvement from image BCE.",
        ],
        "evaluation_device": str(engine.device), "evaluation_batch_size": batch_size,
    })
    metrics.update(checkpoint_provenance(checkpoint_path))
    if metrics["checkpoint_sha256"] != engine.checkpoint_sha256:
        raise ValueError("Checkpoint changed during evaluation")
    metrics["evaluation_additional_source_sha256"] = additional_sources()
    metrics["uncertainty"]["auroc_method"] = "percentile stratified bootstrap of test images; independence is an assumption"
    metrics["uncertainty"]["limitations"] += (
        " KSDD2 product/batch independence is not supplied. These intervals do not correct for prior test inspection."
    )
    return metrics
