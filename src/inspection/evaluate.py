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


def compute_metrics(labels, scores, masks, anomaly_maps, threshold: float, include_pixel_metric: bool = True) -> dict:
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
            model_map = np.asarray(anomaly_maps[index], dtype=np.float32).squeeze(axis=0)
            prediction = np.asarray(Image.fromarray(model_map).resize(
                (width, height), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
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


def _write_overlays(records, maps, scores, threshold, output: Path, limit: int = 12) -> None:
    """Prioritize mistakes, then representative high-scoring correct decisions."""
    output.mkdir(parents=True, exist_ok=True)
    indices = sorted(range(len(records)), key=lambda i: (
        int((scores[i] >= threshold) != bool(records[i]["label"])), float(scores[i])), reverse=True)[:limit]
    manifest = []
    for index in indices:
        record = records[index]
        heatmap = maps[index]
        if heatmap.ndim == 3:
            heatmap = heatmap[0]
        with Image.open(record["image"]) as image:
            image = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
            rgb = np.asarray(image, dtype=np.float32)
        color = np.zeros_like(rgb)
        color[:, :, 0] = 255
        color[:, :, 1] = 48
        alpha = np.clip(heatmap, 0, 1)[:, :, None] * 0.65
        overlay = np.clip(rgb * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
        canvas = np.concatenate((rgb.astype(np.uint8), overlay), axis=1)
        name = f"{index:04d}_{record['category']}_{record['defect']}.png"
        Image.fromarray(canvas).save(output / name)
        manifest.append({"file": name, "source": record["image"], "label": record["label"],
                         "score": float(scores[index]), "predicted_defective": bool(scores[index] >= threshold)})
    (output / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")


def evaluate_checkpoint(manifest_path: Path, checkpoint_path: Path, device_name: str = "auto",
                        overlay_directory: Path | None = None, batch_size: int = 8,
                        pixel_space: str = "original") -> dict:
    import torch
    from torch.utils.data import DataLoader
    from .model import InspectionModel, anomaly_score

    manifest = load_manifest(manifest_path, verify_splits=("test",))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("split_digest") != manifest["split_digest"]:
        raise ValueError("Checkpoint and evaluation manifest differ; use the training manifest")
    if "threshold" not in checkpoint:
        raise ValueError("Checkpoint lacks a threshold fitted on separate normal calibration data")
    config = checkpoint.get("config", {})
    image_size = int(checkpoint.get("image_size", config.get("image_size", manifest["config"]["image_size"])))
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model_config = checkpoint.get("model_config", {"base_channels": config.get("base_channels", 16)})
    model = InspectionModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    records = manifest["splits"]["test"]
    if not records:
        raise ValueError("The original test split is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    dataset = InspectionDataset(records, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    maps, scores, labels, masks, batch_times, single_times = [], [], [], [], [], []
    with torch.inference_mode():
        warmup = dataset[0]["image"][None].to(device)
        for _ in range(3):
            model(warmup)
        _sync(torch, device)
        for batch in loader:
            images = batch["image"].to(device)
            _sync(torch, device)
            start = time.perf_counter()
            _, logits = model(images)
            image_scores = anomaly_score(logits)
            _sync(torch, device)
            batch_times.append((time.perf_counter() - start) * 1000)
            maps.append(torch.sigmoid(logits).cpu().numpy())
            scores.extend(image_scores.cpu().tolist())
            labels.extend(batch["label"].tolist())
            masks.append(batch["mask"].numpy())
        for index in range(min(32, len(dataset))):
            image = dataset[index]["image"][None].to(device)
            _sync(torch, device)
            start = time.perf_counter()
            _, logits = model(image)
            anomaly_score(logits)
            _sync(torch, device)
            single_times.append((time.perf_counter() - start) * 1000)
    maps_array = np.concatenate(maps, axis=0)
    masks_array = np.concatenate(masks, axis=0)
    threshold = float(checkpoint["threshold"])
    if pixel_space not in {"original", "model"}:
        raise ValueError("pixel_space must be original or model")
    metrics = compute_metrics(labels, scores, masks_array, maps_array, threshold,
                              include_pixel_metric=pixel_space == "model")
    if pixel_space == "original":
        metrics.update(original_pixel_metrics(records, maps_array))
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
    metrics["threshold_source"] = checkpoint.get("threshold_source", "checkpoint normal calibration split")
    metrics["score_definition"] = "mean of top 1% sigmoid anomaly-map pixels; not a calibrated probability"
    metrics["limitations"] = [
        "Predictions originate at model resolution; upsampling cannot recover defects lost during input resizing." if pixel_space == "original" else
        "Pixel metrics use resized masks at model resolution and cannot be compared directly with native-resolution benchmarks.",
        "A threshold fitted to a small normal calibration set has uncertain false-alarm generalization.",
        "Synthetic-anomaly training does not guarantee transfer to all real defect types.",
    ]
    metrics["predictions"] = [{"image": record["image"], "category": record["category"],
                               "defect": record["defect"], "label": record["label"],
                               "score": float(score), "predicted_defective": bool(score >= threshold)}
                              for record, score in zip(records, scores)]
    if overlay_directory is not None:
        _write_overlays(records, maps_array, scores, threshold, overlay_directory)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--overlays", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--pixel-space", default="original", choices=["original", "model"])
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.manifest, args.checkpoint, args.device, args.overlays, args.batch_size, args.pixel_space)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in metrics.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
