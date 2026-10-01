"""Describe procedural corruption coverage using normal training images only."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from inspection.data import load_manifest
from inspection.synthesis import synthesize


def quantiles(values: list[float]) -> dict[str, float | None]:
    return {str(q): float(np.quantile(values, q)) if values else None
            for q in (0.05, 0.5, 0.95)}


def audit(manifest_path: Path, category: str = "metal_nut", image_size: int = 128,
          seed: int = 420000, foreground_threshold: float = 0.12,
          resampling: str = "bilinear") -> dict:
    manifest = load_manifest(manifest_path)
    records = [r for r in manifest["splits"]["train"] if r["category"] == category]
    if not records or any(r["label"] != 0 for r in records):
        raise ValueError("Audit requires a nonempty normal-only training subset")
    if image_size < 16 or not 0 < foreground_threshold < 1:
        raise ValueError("Invalid image size or heuristic foreground threshold")
    modes = ("scratch", "appearance", "texture")
    measurements: dict[str, list[float]] = {
        "border_brightness": [], "candidate_foreground_brightness": [],
        "candidate_foreground_image_fraction": [],
        "synthetic_mask_fraction_outside_candidate_foreground": [],
        "synthetic_mask_fraction_of_image": [],
        "absolute_mean_rgb_change_inside_mask": [],
    }
    resize = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}[resampling]
    signature = inspect.signature(synthesize)
    unrestricted_kwargs = {"strategy": "legacy"} if "strategy" in signature.parameters else {}
    for index, record in enumerate(records):
        path = Path(record["image"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError(f"Normal training image changed since manifest audit: {path}")
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB").resize((image_size, image_size), resize),
                             dtype=np.float32).copy() / 255
        brightness = rgb.mean(axis=2)
        foreground = brightness > foreground_threshold
        # Preserve the original descriptive study's border sampling exactly.
        border = np.concatenate((brightness[:5].ravel(), brightness[-5:].ravel(),
                                 brightness[:, :5].ravel(), brightness[:, -5:].ravel()))
        measurements["border_brightness"].extend(border[::8].tolist())
        measurements["candidate_foreground_brightness"].extend(brightness[foreground][::30].tolist())
        measurements["candidate_foreground_image_fraction"].append(float(foreground.mean()))
        clean = torch.from_numpy(rgb).permute(2, 0, 1)
        for mode_index, mode in enumerate(modes):
            generator = torch.Generator().manual_seed(seed + index * len(modes) + mode_index)
            corrupted, mask_tensor = synthesize(clean, generator, mode=mode, **unrestricted_kwargs)
            mask = mask_tensor[0].numpy() > 0
            if not mask.any():
                raise ValueError(f"Forced {mode} corruption produced an empty mask")
            measurements["synthetic_mask_fraction_outside_candidate_foreground"].append(
                float((mask & ~foreground).sum() / mask.sum()))
            measurements["synthetic_mask_fraction_of_image"].append(float(mask.mean()))
            difference = (corrupted - clean).abs().mean(dim=0).numpy()
            measurements["absolute_mean_rgb_change_inside_mask"].extend(difference[mask][::8].tolist())
    synthesis_path = Path(inspect.getfile(synthesize))
    return {
        "category": category, "split": "train", "training_images": len(records),
        "synthetic_examples": len(records) * len(modes),
        "split_digest": manifest["split_digest"], "image_size": image_size,
        "resampling": resampling, "device": "cpu", "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__), "pillow_version": Image.__version__,
        "seed": seed, "seed_rule": "seed + image_index * 3 + mode_index",
        "forced_modes": list(modes), "synthesis_strategy": "legacy", "foreground_only": False,
        "synthesis_source_sha256": hashlib.sha256(synthesis_path.read_bytes()).hexdigest(),
        "candidate_foreground_rule": f"mean RGB > {foreground_threshold}",
        "brightness_definition": "arithmetic mean RGB in [0,1], not perceptual luminance",
        "sampling": {"border": "first/last five rows and columns, every eighth value",
                     "foreground_brightness": "every thirtieth candidate foreground pixel per image",
                     "absolute_change": "every eighth masked pixel per synthetic image"},
        "quantiles": {name: quantiles(values) for name, values in measurements.items()},
        "measurement_samples": {name: len(values) for name, values in measurements.items()},
        "limitations": [
            "The foreground is a brightness heuristic, not an annotated segmentation.",
            "Only original normal training records are read; validation, calibration and test images are not read.",
            "Forced corruption modes omit the generator's probability of returning a clean example.",
            "These descriptive measurements do not establish a cause of real-defect failures.",
            "The default bilinear resizing matches the training loader; bicubic is an optional descriptive comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=420000)
    parser.add_argument("--foreground-threshold", type=float, default=0.12)
    parser.add_argument("--resampling", choices=("bicubic", "bilinear"), default="bilinear")
    args = parser.parse_args()
    report = audit(args.manifest, args.category, args.image_size, args.seed,
                   args.foreground_threshold, args.resampling)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"training_images": report["training_images"],
                      "synthetic_examples": report["synthetic_examples"],
                      "quantiles": report["quantiles"]}, indent=2))


if __name__ == "__main__":
    main()
