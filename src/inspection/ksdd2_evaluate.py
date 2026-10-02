"""Frozen supervised KSDD2 evaluation with original masks and letterbox restoration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import torch

from .checkpointing import atomic_json
from .ksdd2 import load_manifest
from .preprocessing import prepare_image, restore_map


def native_pixel_metrics(records, model_maps, preprocess):
    """Crop letterbox padding before resize; target masks are never resized."""
    if len(records) != len(model_maps) or not len(records):
        raise ValueError("Nonempty aligned image records and maps are required")
    targets, predictions = [], []
    for record, model_map in zip(records, model_maps):
        with Image.open(record["image"]) as image:
            _, _, transform = prepare_image(image.convert("RGB"), preprocess)
        native_map = restore_map(model_map, transform)
        with Image.open(record["mask"]) as mask:
            target = np.asarray(mask) > 0
        if target.shape != native_map.shape or not np.isfinite(native_map).all():
            raise ValueError("Native prediction and original mask must align and be finite")
        targets.append(target.ravel())
        predictions.append(native_map.ravel())
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    return {"pixel_average_precision": float(average_precision_score(target, prediction)) if target.any() else None,
            "pixel_metric_space": "original", "pixel_aggregation": "pooled pixels in one KSDD2 surface category",
            "pixel_resize": "crop letterbox padding, bilinear restore predictions, unchanged original masks",
            "pixels": int(target.size), "positive_pixels": int(target.sum())}


def evaluate(manifest_path, checkpoint_path, device="auto", batch_size=8, gallery=None,
             bootstrap_samples=1000, experimental_status="frozen-held-out"):
    from .evaluate import evaluate_checkpoint
    manifest = load_manifest(manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != "supervised_segmentation":
        raise ValueError("KSDD2 supervised evaluation requires supervised_segmentation weights")
    if checkpoint.get("split_digest") != manifest["split_digest"]:
        raise ValueError("Supervised checkpoint and audited KSDD2 split differ")
    preprocess = checkpoint.get("preprocess", {})
    if preprocess.get("mode") != "letterbox":
        raise ValueError("Supervised KSDD2 checkpoint must declare letterbox preprocessing")
    if experimental_status not in {"exploratory", "frozen-held-out"}:
        raise ValueError("Unknown experimental status")
    metrics = evaluate_checkpoint(Path(manifest_path), Path(checkpoint_path), device,
                                  Path(gallery) if gallery else None, batch_size, "original",
                                  bootstrap_samples, experimental_status)
    metrics.update({"training_supervision": "real defective and normal training images with pixel annotations",
                    "comparison_scope": "separate supervised task; not directly comparable to normal-only MVTec",
                    "validation_selection": checkpoint.get("selection_rule"),
                    "validation_best_epoch": checkpoint.get("best_epoch"),
                    "validation_best_metric": checkpoint.get("best_metric"),
                    "training_seed": checkpoint.get("config", {}).get("training_seed"),
                    "calibration_normal_count": checkpoint.get("calibration", {}).get("normal_count"),
                    "pixel_resize": "crop letterbox padding, bilinear restore predictions, unchanged original masks",
                    "product_group_metadata": "not supplied; image-similarity checks do not establish batch/product independence",
                    "limitations": [
                        "Real-defect supervision does not establish recognition of previously unseen defect families.",
                        "Restored maps cannot recover visual details lost during input resampling.",
                        "Calibration quantiles do not guarantee future false-alarm rates.",
                        "Provided image identities do not establish independent physical products or acquisition batches.",
                    ]})
    metrics["uncertainty"]["auroc_method"] = "percentile stratified bootstrap of test images; independence is an assumption"
    metrics["uncertainty"]["limitations"] += " KSDD2 product/batch independence is not supplied."
    if gallery:
        directory = Path(gallery)
        index = json.loads((directory / "index.json").read_text())
        index["license"] = "KolektorSDD2 derivatives: CC BY-NC-SA 4.0"
        for item in index["items"]:
            matches = [record for record in manifest["splits"]["test"]
                       if Path(record["image"]).name == Path(item["source"]).name]
            if len(matches) == 1:
                item["source"] = matches[0]["relative_image"]
        atomic_json(directory / "index.json", index)
        (directory / "ATTRIBUTION.md").write_text(
            "Images and annotations are derived from [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/), "
            "provided by Kolektor Group and published by Jakob Božič, Domen Tabernik and Danijel Skočaj. "
            "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
            "Changes: model activation overlays, ground-truth overlays, panel composition and labels. "
            "These derivative images retain the dataset license.\n")
        metrics["gallery"] = index
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/ksdd2-manifest.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlays", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--experimental-status", choices=("frozen-held-out", "exploratory"), default="frozen-held-out")
    args = parser.parse_args()
    metrics = evaluate(args.manifest, args.checkpoint, args.device, args.batch_size, args.overlays,
                       args.bootstrap_samples, args.experimental_status)
    atomic_json(args.output, metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
