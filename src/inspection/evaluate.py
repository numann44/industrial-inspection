"""Evaluate a frozen checkpoint on the untouched original MVTec AD test set."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import InspectionDataset, load_manifest
from .reporting import checkpoint_provenance, defect_area_groups, image_uncertainty, write_failure_gallery


def load_evaluation_manifest(path, verify_splits=("test",)):
    """Dispatch audited dataset schemas without reopening optimization splits."""
    metadata = json.loads(Path(path).read_text())
    if metadata.get("dataset") == "KolektorSDD2":
        from .ksdd2 import load_manifest as load_ksdd2_manifest
        return load_ksdd2_manifest(path, verify_splits=verify_splits)
    if metadata.get("dataset") not in {None, "MVTec AD"}:
        raise ValueError("Unsupported evaluation dataset schema")
    return load_manifest(path, verify_splits=verify_splits)


def compute_metrics(labels, scores, masks, anomaly_maps, threshold: float, include_pixel_metric: bool = True,
                    valid_masks=None) -> dict:
    """The operating threshold is an input, never selected using test labels."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks).astype(bool)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    if not len(labels) or labels.shape != scores.shape or masks.shape != anomaly_maps.shape:
        raise ValueError("Nonempty labels/scores and aligned masks/maps are required")
    if masks.shape[0] != len(labels) or not np.isfinite(scores).all() or not np.isfinite(anomaly_maps).all():
        raise ValueError("Predictions must be finite and aligned with images")
    if not np.isfinite(threshold) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Threshold must be finite and labels binary")
    predictions = scores >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    pixel_labels = masks.ravel()
    pixel_scores = anomaly_maps.ravel()
    if valid_masks is not None:
        valid = np.asarray(valid_masks, dtype=bool)
        if valid.shape != masks.shape or not valid.any():
            raise ValueError("Valid masks must align with masks/maps and contain actual image pixels")
        pixel_labels, pixel_scores = masks[valid], anomaly_maps[valid]
    return {
        "image_auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "image_average_precision": float(average_precision_score(labels, scores)) if labels.any() else None,
        "pixel_average_precision": float(average_precision_score(pixel_labels, pixel_scores)) if include_pixel_metric and pixel_labels.any() else None,
        "threshold": float(threshold),
        "decision_rule": "anomaly_score >= threshold",
        "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "defect_recall": tp / (tp + fn) if tp + fn else None,
        "normal_false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "images": len(labels),
        "evaluation_resolution": list(masks.shape[-2:]),
    }


def original_pixel_metrics(records: list[dict], anomaly_maps: np.ndarray) -> dict:
    """Upsample predictions, preserve native masks, and release each category's arrays."""
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["category"]].append(index)
    results = {}
    for category, indices in sorted(groups.items()):
        target_chunks, prediction_chunks = [], []
        for index in indices:
            record = records[index]
            width, height = record["original_size"]
            if record["mask"] is not None:
                with Image.open(record["mask"]) as mask:
                    target = np.asarray(mask).astype(bool).ravel()
            else:
                target = np.zeros(width * height, dtype=bool)
            model_map = np.asarray(anomaly_maps[index], dtype=np.float32)
            if model_map.ndim == 3:
                model_map = model_map.squeeze(axis=0)
            if model_map.ndim != 2:
                raise ValueError("Expected a 2D map per image")
            prediction = (model_map if model_map.shape == (height, width) else np.asarray(Image.fromarray(model_map).resize(
                (width, height), Image.Resampling.BILINEAR), dtype=np.float32)).ravel()
            target_chunks.append(target)
            prediction_chunks.append(prediction)
        targets = np.concatenate(target_chunks)
        predictions = np.concatenate(prediction_chunks)
        del target_chunks, prediction_chunks
        results[category] = {"average_precision": float(average_precision_score(targets, predictions)) if targets.any() else None,
                             "pixels": int(len(targets)), "positive_pixels": int(targets.sum()),
                             "images": len(indices)}
        del targets, predictions
    values = [result["average_precision"] for result in results.values() if result["average_precision"] is not None]
    return {"pixel_average_precision": float(np.mean(values)) if values else None,
            "pixel_average_precision_by_category": results,
            "pixel_metric_space": "original",
            "pixel_aggregation": "macro mean of per-category pixel average precision",
            "pixel_resize": "bilinear anomaly-map upsampling; original masks are unchanged"}


def _sync(torch, device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_checkpoint(manifest_path: Path, checkpoint_path: Path, device_name: str = "auto",
                        overlay_directory: Path | None = None, batch_size: int = 8,
                        pixel_space: str = "original", bootstrap_samples: int = 1000,
                        experimental_status: str = "exploratory") -> dict:
    import torch
    from torch.utils.data import DataLoader
    from .engine import InspectionEngine
    from .preprocessing import decode_image, prepare_image, restore_map

    manifest = load_evaluation_manifest(manifest_path, verify_splits=("test",))
    engine = InspectionEngine(checkpoint_path, device=device_name)
    checkpoint = engine.checkpoint
    if checkpoint.get("split_digest") != manifest["split_digest"]:
        raise ValueError("Checkpoint and evaluation manifest differ; use the training manifest")
    if "threshold" not in checkpoint:
        raise ValueError("Checkpoint lacks a threshold fitted on separate normal calibration data")
    device = engine.device
    records = manifest["splits"]["test"]
    if not records:
        raise ValueError("The original test split is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(record["category"] != engine.category for record in records):
        raise ValueError("All evaluation images must match the checkpoint category")
    if manifest.get("dataset") == "KolektorSDD2":
        from .ksdd2 import KSDD2Dataset
        dataset = KSDD2Dataset(records, preprocess=engine.preprocess)
    else:
        dataset = InspectionDataset(records, engine.image_size, preprocess=engine.preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    maps, scores, labels, masks, valid_masks, batch_times, single_times = [], [], [], [], [], [], []
    with torch.inference_mode():
        warmup = dataset[0]["image"][None].to(device)
        warmup_valid = dataset[0]["valid_mask"][None].to(device)
        for _ in range(3):
            engine.predict_tensor(warmup, warmup_valid)
        _sync(torch, device)
        for batch in loader:
            images = batch["image"].to(device)
            _sync(torch, device)
            start = time.perf_counter()
            activation, image_scores = engine.predict_tensor(images, batch["valid_mask"])
            _sync(torch, device)
            batch_times.append((time.perf_counter() - start) * 1000)
            maps.append(activation.cpu().numpy())
            scores.extend(image_scores.cpu().tolist())
            labels.extend(batch["label"].tolist())
            masks.append(batch["mask"].numpy())
            valid_masks.append(batch["valid_mask"].numpy())
        for index in range(min(32, len(dataset))):
            sample = dataset[index]
            image = sample["image"][None].to(device)
            valid = sample["valid_mask"][None].to(device)
            _sync(torch, device)
            start = time.perf_counter()
            engine.predict_tensor(image, valid)
            _sync(torch, device)
            single_times.append((time.perf_counter() - start) * 1000)
    maps_array = np.concatenate(maps, axis=0)
    masks_array = np.concatenate(masks, axis=0)
    valid_array = np.concatenate(valid_masks, axis=0)
    native_maps = []
    if pixel_space == "original" or overlay_directory is not None:
        for index, record in enumerate(records):
            image = decode_image(record["image"])
            _, _, transform = prepare_image(image, engine.preprocess)
            native_maps.append(restore_map(maps_array[index, 0], transform))
    threshold = engine.threshold
    if pixel_space not in {"original", "model"}:
        raise ValueError("pixel_space must be original or model")
    metrics = compute_metrics(labels, scores, masks_array, maps_array, threshold,
                              include_pixel_metric=pixel_space == "model", valid_masks=valid_array)
    if pixel_space == "original":
        metrics.update(original_pixel_metrics(records, native_maps))
        metrics["evaluation_resolution"] = "original resolution per image"
    else:
        metrics["pixel_metric_space"] = "model"
        metrics["pixel_aggregation"] = "pooled pixels at model resolution"
    defect_groups = defaultdict(list)
    for record, score in zip(records, scores):
        if record["label"]:
            defect_groups[f"{record['category']}/{record['defect']}"].append(score >= threshold)
    metrics["recall_by_defect"] = {
        name: {"recall": float(np.mean(values)), "images": len(values), "detected": int(sum(values))}
        for name, values in sorted(defect_groups.items())}
    metrics["latency"] = {
        "single_image_median_ms": float(np.median(single_times)),
        "single_image_p95_ms": float(np.quantile(single_times, 0.95)),
        "single_image_measurements": len(single_times),
        "batch_median_ms": float(np.median(batch_times)), "batch_p95_ms": float(np.quantile(batch_times, 0.95)),
        "batch_size": batch_size, "batches": len(batch_times), "warmup_forwards": 3,
        "scope": "model forward and image scoring only; excludes disk loading and host-to-device transfer",
        "device": str(device),
    }
    metrics["manifest_digest"] = manifest["digest"]
    metrics["split_digest"] = manifest["split_digest"]
    metrics["checkpoint"] = str(checkpoint_path.resolve())
    metrics.update(checkpoint_provenance(checkpoint_path))
    metrics["model_sha256"] = engine.model_sha256
    metrics["model_kind"] = engine.model_kind
    metrics["preprocess"] = engine.preprocess
    metrics["experimental_status"] = experimental_status
    metrics["dataset"] = manifest.get("dataset", "MVTec AD")
    metrics["threshold_source"] = checkpoint.get("threshold_source", "checkpoint normal calibration split")
    metrics["score_definition"] = engine.score_definition
    metrics["uncertainty"] = image_uncertainty(labels, scores, threshold, bootstrap_samples=bootstrap_samples)
    metrics["defect_area_groups"] = defect_area_groups(records, scores, threshold)
    metrics["class_prevalence"] = {"normal": int(np.sum(np.asarray(labels) == 0)),
                                   "defective": int(np.sum(np.asarray(labels) == 1)),
                                   "defective_fraction": float(np.mean(labels))}
    metrics["limitations"] = [
        "Predictions originate at model resolution; upsampling cannot recover defects lost during input resizing." if pixel_space == "original" else
        "Pixel metrics use resized masks at model resolution and cannot be compared directly with native-resolution benchmarks.",
        "A threshold fitted to a small normal calibration set has uncertain false-alarm generalization.",
        "Dataset training does not guarantee transfer to new defect families, cameras or production conditions.",
    ]
    metrics["predictions"] = [{"image": record["image"], "category": record["category"],
                               "defect": record["defect"], "label": record["label"],
                               "score": float(score), "predicted_defective": bool(score >= threshold)}
                              for record, score in zip(records, scores)]
    if overlay_directory is not None:
        metrics["gallery"] = write_failure_gallery(records, native_maps, scores, threshold, overlay_directory,
                                                    display_max=engine.display_max)
    return metrics


def stress_test_checkpoint(manifest_path: Path, checkpoint_path: Path, device_name: str = "auto",
                           batch_size: int = 8) -> dict:
    """Paired, predeclared image-level stress test; no retraining or threshold tuning."""
    import io
    import torch
    from PIL import ImageEnhance, ImageFilter
    from .engine import InspectionEngine
    from .preprocessing import decode_image, prepare_image

    manifest = load_evaluation_manifest(manifest_path, verify_splits=("test",))
    engine = InspectionEngine(checkpoint_path, device_name)
    if engine.checkpoint.get("split_digest") != manifest["split_digest"]:
        raise ValueError("Stress evaluation must use the checkpoint's original audited split")
    records = manifest["splits"]["test"]
    if batch_size < 1 or not records or any(record["category"] != engine.category for record in records):
        raise ValueError("Nonempty category-compatible records and positive batch size required")

    def compressed(image):
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=60)
        return decode_image(stream.getvalue())

    perturbations = {"original": lambda image: image,
                     "gaussian_blur_radius_1": lambda image: image.filter(ImageFilter.GaussianBlur(1)),
                     "brightness_0.8": lambda image: ImageEnhance.Brightness(image).enhance(0.8),
                     "brightness_1.2": lambda image: ImageEnhance.Brightness(image).enhance(1.2),
                     "jpeg_quality_60": compressed}
    labels = [record["label"] for record in records]
    results, reference_scores = {}, None
    for name, perturb in perturbations.items():
        scores = []
        for start in range(0, len(records), batch_size):
            prepared = [prepare_image(perturb(decode_image(record["image"])), engine.preprocess)
                        for record in records[start:start + batch_size]]
            images = torch.stack([item[0] for item in prepared])
            valid = torch.stack([item[1] for item in prepared])
            _, batch_scores = engine.predict_tensor(images, valid)
            scores.extend(batch_scores.cpu().tolist())
        placeholders = np.zeros((len(labels), 1, 1, 1), dtype=np.float32)
        metrics = compute_metrics(labels, scores, placeholders, placeholders, engine.threshold, include_pixel_metric=False)
        metrics.pop("evaluation_resolution")
        metrics.pop("pixel_average_precision")
        if reference_scores is None:
            reference_scores = np.asarray(scores)
        reference_decisions = reference_scores >= engine.threshold
        decisions = np.asarray(scores) >= engine.threshold
        metrics.update({"decision_flips": int(np.sum(reference_decisions != decisions)),
                        "accepted_to_flagged": int(np.sum(~reference_decisions & decisions)),
                        "flagged_to_accepted": int(np.sum(reference_decisions & ~decisions)),
                        "mean_absolute_score_change": float(np.mean(np.abs(reference_scores - scores)))})
        results[name] = metrics
    return {"status": "exploratory paired dataset stress test", "category": engine.category,
            "model_kind": engine.model_kind, "split_digest": manifest["split_digest"],
            "checkpoint_sha256": engine.checkpoint_sha256, "threshold": engine.threshold,
            "threshold_source": "frozen checkpoint calibration; unchanged for every perturbation",
            "results": results, "pixel_metrics": "not computed: image decision stress test only",
            "limitations": "Dataset perturbations do not establish reliability under unseen cameras or production conditions."}


def benchmark_checkpoint(checkpoint_path: Path, image_path: Path, device_name: str = "cpu",
                         measurements: int = 32, warmup: int = 3) -> dict:
    """Measure one frozen model on one normal example after other jobs finish."""
    if measurements < 1 or warmup < 1:
        raise ValueError("Measurement and warmup counts must be positive")
    from .engine import InspectionEngine, synchronize
    from .preprocessing import decode_image, prepare_image
    engine = InspectionEngine(checkpoint_path, device=device_name)
    original = decode_image(image_path)
    tensor, valid, _ = prepare_image(original, engine.preprocess)
    image, valid = tensor[None].to(engine.device), valid[None].to(engine.device)
    for _ in range(warmup):
        engine.predict_tensor(image, valid)
    synchronize(engine.device)
    timings = []
    for _ in range(measurements):
        synchronize(engine.device)
        started = time.perf_counter()
        engine.predict_tensor(image, valid)
        synchronize(engine.device)
        timings.append((time.perf_counter() - started) * 1000)
    return {"device": str(engine.device), "model_kind": engine.model_kind,
            "checkpoint_sha256": engine.checkpoint_sha256, "model_sha256": engine.model_sha256,
            "preprocess": engine.preprocess, "measurements": measurements, "warmup_forwards": warmup,
            "single_image_median_ms": float(np.median(timings)),
            "single_image_p95_ms": float(np.quantile(timings, 0.95)),
            "scope": "warm forward and scoring only; excludes model loading, decode, preprocessing, transfer and rendering",
            "benchmark_condition": "Run after training and other inference jobs finish; this function cannot enforce system idleness."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--overlays", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pixel-space", default="original", choices=["original", "model"])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--experimental-status", default="exploratory", choices=["exploratory", "frozen-held-out"])
    parser.add_argument("--stress-output", type=Path, help="Optionally write paired, fixed-perturbation image-level stress metrics")
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.manifest, args.checkpoint, args.device, args.overlays, args.batch_size,
                                  args.pixel_space, args.bootstrap_samples, args.experimental_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    if args.stress_output is not None:
        stress = stress_test_checkpoint(args.manifest, args.checkpoint, args.device, args.batch_size)
        args.stress_output.parent.mkdir(parents=True, exist_ok=True)
        args.stress_output.write_text(json.dumps(stress, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
