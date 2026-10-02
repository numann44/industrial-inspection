"""Audit MVTec AD and keep its original test set sealed from training."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


def _digest(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key not in {"digest", "split_digest"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _split_digest(manifest: dict) -> str:
    root = Path(manifest["root"])
    config = manifest["config"]
    payload = {"seed": config["seed"], "validation_fraction": config["validation_fraction"],
               "calibration_fraction": config["calibration_fraction"], "splits": {}}
    for split, records in manifest["splits"].items():
        payload["splits"][split] = [
            {"image": Path(record["image"]).relative_to(root).as_posix(),
             "mask": Path(record["mask"]).relative_to(root).as_posix() if record["mask"] else None,
             **{key: record[key] for key in ("category", "defect", "label", "image_sha256", "mask_sha256")}}
            for record in records]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: str | Path, verify_splits: tuple[str, ...] | list[str] = ()) -> dict:
    """Verify metadata and optionally rehash specified splits without reading others."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("digest") != _digest(manifest):
        raise ValueError("Manifest digest mismatch: regenerate the audited manifest after changes")
    if manifest.get("split_digest") != _split_digest(manifest):
        raise ValueError("Manifest split digest mismatch")
    for split in verify_splits:
        for record in manifest["splits"][split]:
            for key in ("image", "mask"):
                if record[key] is not None:
                    source = Path(record[key])
                    if hashlib.sha256(source.read_bytes()).hexdigest() != record[f"{key}_sha256"]:
                        raise ValueError(f"Audited content changed: {source}")
    return manifest


def _inspect_image(path: Path, *, mask: bool = False) -> tuple[tuple[int, int], str]:
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
            if mask:
                array = np.asarray(image)
                if array.ndim != 2 or not set(np.unique(array)).issubset({0, 255}):
                    raise ValueError(f"Mask must be a binary grayscale PNG (0/255): {path}")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"Cannot decode PNG: {path}") from exc
    return size, hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    root: str | Path, categories: list[str] | tuple[str, ...] = ("metal_nut",),
    seed: int = 42, validation_fraction: float = 0.15, calibration_fraction: float = 0.15, image_size: int = 128,
    archive_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Decode every selected PNG, verify masks, split only original train/good."""
    root = Path(root).resolve()
    if not 0 < validation_fraction < 1 or not 0 < calibration_fraction < 1 or validation_fraction + calibration_fraction >= 1:
        raise ValueError("validation and calibration fractions must be positive with sum below 1")
    if image_size < 16:
        raise ValueError("image_size must be at least 16")
    categories = sorted(set(categories))
    if not categories or any(category not in CATEGORIES for category in categories):
        raise ValueError(f"Choose MVTec AD categories from {CATEGORIES}")
    if archive_sha256 is not None and (
        len(archive_sha256) != 64 or any(char not in "0123456789abcdef" for char in archive_sha256.lower())
    ):
        raise ValueError("archive_sha256 must be a 64-character SHA-256 hex digest")

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "calibration": [], "test": []}
    sizes: Counter = Counter()
    vanished_masks: Counter = Counter()
    decoded = 0
    for category in categories:
        directory = root / category
        train_directory = directory / "train" / "good"
        training_paths = sorted(train_directory.glob("*.png"))
        if len(training_paths) < 3:
            raise ValueError(f"Need at least three normal training PNGs: {train_directory}")
        extra_train = [p for p in (directory / "train").glob("*/*.png") if p.parent.name != "good"]
        if extra_train:
            raise ValueError(f"Original MVTec AD training must contain only good images: {extra_train[0]}")
        test_paths = sorted((directory / "test").glob("*/*.png"))
        if not test_paths or not any(p.parent.name == "good" for p in test_paths):
            raise ValueError(f"Missing original test set or good test images: {directory / 'test'}")
        if not any(p.parent.name != "good" for p in test_paths):
            raise ValueError(f"Missing defective test images: {directory / 'test'}")

        shuffled = list(training_paths)
        random.Random(f"{seed}:{category}").shuffle(shuffled)
        count = max(1, min(len(shuffled) - 2, round(len(shuffled) * validation_fraction)))
        calibration_count = max(1, min(len(shuffled) - count - 1, round(len(shuffled) * calibration_fraction)))
        validation_paths = set(shuffled[:count])
        calibration_paths = set(shuffled[count:count + calibration_count])
        referenced_masks: set[Path] = set()
        for path in training_paths + test_paths:
            size, image_hash = _inspect_image(path)
            decoded += 1
            sizes[f"{size[0]}x{size[1]}"] += 1
            is_test = path in test_paths
            defect = path.parent.name if is_test else "good"
            mask_path = None
            mask_hash = None
            if defect != "good":
                mask_path = directory / "ground_truth" / defect / f"{path.stem}_mask.png"
                if not mask_path.is_file():
                    raise ValueError(f"Missing ground-truth mask: {mask_path}")
                mask_size, mask_hash = _inspect_image(mask_path, mask=True)
                decoded += 1
                if size != mask_size:
                    raise ValueError(f"Image/mask dimensions differ: {path}, {mask_path}")
                with Image.open(mask_path) as mask:
                    if not np.any(np.asarray(mask)):
                        raise ValueError(f"Defective image has an empty mask: {mask_path}")
                    if not np.any(np.asarray(mask.resize((image_size, image_size), Image.Resampling.NEAREST))):
                        vanished_masks[f"{category}/{defect}"] += 1
                referenced_masks.add(mask_path)
            record = {"image": str(path), "mask": str(mask_path) if mask_path else None,
                      "category": category, "defect": defect, "label": int(defect != "good"),
                      "image_sha256": image_hash, "mask_sha256": mask_hash, "original_size": list(size)}
            split = "test" if is_test else "validation" if path in validation_paths else "calibration" if path in calibration_paths else "train"
            splits[split].append(record)
        all_masks = set((directory / "ground_truth").glob("*/*.png"))
        orphaned = all_masks - referenced_masks
        if orphaned:
            raise ValueError(f"Unmatched ground-truth mask: {sorted(orphaned)[0]}")
        known_pngs = set(training_paths + test_paths) | all_masks
        unexpected = set(directory.rglob("*.png")) - known_pngs
        if unexpected:
            raise ValueError(f"Unexpected unaudited PNG inside category: {sorted(unexpected)[0]}")

    hash_locations: dict[str, set[str]] = defaultdict(set)
    for split, records in splits.items():
        for record in records:
            hash_locations[record["image_sha256"]].add(split)
    leaked = {digest: sorted(locations) for digest, locations in hash_locations.items() if len(locations) > 1}
    if leaked:
        raise ValueError(f"Identical image bytes cross train/validation/calibration/test boundaries: {leaked}")

    counts = {
        split: {"total": len(records), "by_category_defect": dict(sorted(Counter(
            f"{record['category']}/{record['defect']}" for record in records).items()))}
        for split, records in splits.items()
    }
    manifest = {"schema_version": 1, "root": str(root),
                "config": {"seed": seed, "validation_fraction": validation_fraction, "calibration_fraction": calibration_fraction,
                           "image_size": image_size, "categories": categories},
                "provenance": {"dataset": "MVTec AD", "archive_sha256": archive_sha256},
                "splits": splits, "counts": counts}
    manifest["split_digest"] = _split_digest(manifest)
    manifest["digest"] = _digest(manifest)
    audit = {"status": "passed", "manifest_digest": manifest["digest"],
             "decoded_pngs": decoded, "counts": counts, "image_sizes": dict(sorted(sizes.items())),
             "cross_split_duplicate_hashes": 0, "archive_sha256": archive_sha256,
             "masks_vanishing_after_resize": {"image_size": image_size, "total": sum(vanished_masks.values()),
                                               "by_category_defect": dict(sorted(vanished_masks.items()))},
             "checks": ["all selected PNGs decode", "normal-only training, validation and calibration",
                        "original test membership preserved", "binary masks match image dimensions",
                        "no orphan masks", "no identical image bytes across splits"]}
    return manifest, audit


class InspectionDataset:
    """Small CPU loader: RGB in [0,1], nearest-neighbor binary mask resizing."""
    def __init__(self, records: list[dict], image_size: int = 128, include_masks: bool = True, preprocess=None):
        from .preprocessing import normalize_spec
        self.records = records
        self.image_size = image_size
        self.include_masks = include_masks
        self.preprocess = normalize_spec(preprocess, image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        import torch
        from .preprocessing import decode_image, prepare_image, prepare_mask
        record = self.records[index]
        image = decode_image(record["image"])
        tensor, valid_mask, _ = prepare_image(image, self.preprocess)
        sample = {"image": tensor, "valid_mask": valid_mask,
                  "label": record["label"], "path": record["image"],
                  "category": record["category"], "defect": record["defect"]}
        if self.include_masks:
            if record["mask"]:
                with Image.open(record["mask"]) as image:
                    mask = prepare_mask(image, self.preprocess)
            else:
                mask = torch.zeros((1, self.preprocess["height"], self.preprocess["width"]), dtype=torch.float32)
            sample["mask"] = mask
        return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["metal_nut"], choices=CATEGORIES)
    parser.add_argument("--output", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("data/audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    manifest, audit = build_manifest(args.root, args.categories, seed=args.seed,
                                     image_size=args.image_size, archive_sha256=args.archive_sha256)
    for path, content in ((args.output, manifest), (args.report, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "counts": audit["counts"],
                      "manifest_digest": manifest["digest"]}, indent=2))


if __name__ == "__main__":
    main()
