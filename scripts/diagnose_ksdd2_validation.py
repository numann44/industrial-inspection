"""Planning diagnostic on validation only; never opens calibration/test images."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import ImageEnhance, ImageFilter
import torch

from inspection.checkpointing import atomic_json
from inspection.engine import InspectionEngine
from inspection.evaluate import compute_metrics
from inspection.ksdd2 import load_manifest
from inspection.preprocessing import decode_image, prepare_image
from scripts.run_study import scheduler_lock


def diagnose(manifest_path, checkpoint_path, output, device="cpu"):
    output = Path(output)
    if output.exists():
        raise ValueError("Planning evidence already exists; do not overwrite it")
    manifest = load_manifest(manifest_path, verify_splits=("validation",))
    records = manifest["splits"]["validation"]
    engine = InspectionEngine(checkpoint_path, device)
    if engine.category != "kolektor_surface" or engine.checkpoint["split_digest"] != manifest["split_digest"]:
        raise ValueError("Checkpoint and original surface split must match")

    def jpeg(image):
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=60)
        return decode_image(stream.getvalue())

    conditions = {
        "original": lambda image: image,
        "gaussian_blur_radius_1": lambda image: image.filter(ImageFilter.GaussianBlur(1)),
        "brightness_0.8": lambda image: ImageEnhance.Brightness(image).enhance(.8),
        "brightness_1.2": lambda image: ImageEnhance.Brightness(image).enhance(1.2),
        "jpeg_quality_60": jpeg,
    }
    labels = [r["label"] for r in records]
    results = {}
    for name, transform in conditions.items():
        scores = []
        for start in range(0, len(records), 8):
            prepared = [prepare_image(transform(decode_image(r["image"])), engine.preprocess)
                        for r in records[start:start + 8]]
            _, values = engine.predict_tensor(torch.stack([r[0] for r in prepared]),
                                              torch.stack([r[1] for r in prepared]))
            scores.extend(values.cpu().tolist())
        empty = np.zeros((len(records), 1, 1, 1), dtype=np.float32)
        result = compute_metrics(labels, scores, empty, empty, engine.threshold, include_pixel_metric=False)
        result.pop("evaluation_resolution", None)
        result.pop("pixel_average_precision", None)
        result["predictions"] = [{"image": r["relative_image"], "label": r["label"], "score": s}
                                 for r, s in zip(records, scores)]
        results[name] = result
        print(json.dumps({"condition": name, **{k: v for k, v in result.items() if k != "predictions"}}), flush=True)
    report = {"purpose": "validation-only planning diagnostic, not independent test evidence",
              "split": "validation", "samples": len(records),
              "real_test_images_read": 0, "calibration_images_read": 0,
              "split_digest": manifest["split_digest"],
              "checkpoint_sha256": engine.checkpoint_sha256, "threshold": engine.threshold,
              "threshold_source": "existing immutable v2 checkpoint; no recalibration",
              "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "conditions": results}
    atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/ksdd2-manifest.json")
    parser.add_argument("--checkpoint", default="runs/study-v2-ksdd2-seed44/checkpoint.pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with scheduler_lock(Path(__file__).resolve().parents[1] / "outputs/mps-study.lock"):
        diagnose(args.manifest, args.checkpoint, args.output, args.device)
