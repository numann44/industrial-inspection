"""Immutable, shared synthetic model-selection bank from validation normals only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import InspectionDataset, load_manifest
from .synthesis import synthesize

FAMILIES = (("legacy", "scratch"), ("legacy", "appearance"), ("legacy", "texture"),
            ("foreground", "scratch"), ("foreground", "lighting"), ("foreground", "texture"),
            ("foreground", "warp"), ("foreground", "appearance"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(bank: dict) -> str:
    payload = {key: value for key, value in bank.items() if key != "bank_digest"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_bank(bank_path: str | Path, verify_content: bool = True) -> dict:
    bank_path = Path(bank_path)
    bank = json.loads(bank_path.read_text())
    if bank.get("bank_digest") != _digest(bank):
        raise ValueError("Synthetic bank metadata digest mismatch")
    if bank.get("schema_version") != 1 or bank.get("source_split") != "validation":
        raise ValueError("Expected bank schema 1 sourced exclusively from validation normals")
    if verify_content:
        verified = set()
        for record in bank["records"]:
            for name in ("image", "mask", "clean"):
                relative = Path(record[name])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Bank paths must remain inside the bank directory")
                path = bank_path.parent / relative
                if path not in verified:
                    if _hash(path) != record[f"{name}_sha256"]:
                        raise ValueError(f"Synthetic bank content changed: {relative}")
                    verified.add(path)
    return bank


def create_bank(manifest_path: str | Path, output_dir: str | Path,
                seed: int = 1_000_042, image_size: int = 256) -> dict:
    """Persist canonical float images and masks; never read train/calibration/test."""
    if image_size < 16:
        raise ValueError("Canonical image size must be at least 16")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("A bank directory must be empty; existing immutable banks cannot be overwritten")
    manifest = load_manifest(manifest_path, verify_splits=("validation",))
    records = manifest["splits"]["validation"]
    if not records or any(record["label"] != 0 or record["mask"] is not None for record in records):
        raise ValueError("Bank sources must be nonempty original validation normals")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "arrays").mkdir()
    dataset = InspectionDataset(records, image_size, include_masks=False)
    bank_records = []
    for index, source in enumerate(records):
        clean = dataset[index]["image"]
        clean_path = Path("arrays") / f"{index:04d}_clean.npy"
        np.save(output_dir / clean_path, clean.numpy().astype(np.float32), allow_pickle=False)
        clean_hash = _hash(output_dir / clean_path)
        for family_index, (strategy, mode) in enumerate((("clean", "clean"),) + FAMILIES):
            selected_seed = seed + index * 10_000 + family_index * 100
            attempts = 0
            if strategy == "clean":
                image, mask = clean.clone(), torch.zeros_like(clean[:1])
            else:
                for attempts in range(64):
                    generator = torch.Generator().manual_seed(selected_seed + attempts)
                    image, mask = synthesize(clean, generator, normal_probability=0.0,
                                             strategy=strategy, mode=mode)
                    if mask.any():
                        break
                else:
                    raise ValueError(f"Cannot generate nonempty {strategy}/{mode} anomaly from validation source {index}")
            image_path = clean_path if strategy == "clean" else Path("arrays") / f"{index:04d}_{strategy}_{mode}.npy"
            mask_path = Path("arrays") / f"{index:04d}_{strategy}_{mode}_mask.npy"
            if strategy != "clean":
                np.save(output_dir / image_path, image.numpy().astype(np.float32), allow_pickle=False)
            np.save(output_dir / mask_path, mask.numpy().astype(np.uint8), allow_pickle=False)
            bank_records.append({"image": image_path.as_posix(), "mask": mask_path.as_posix(),
                                 "clean": clean_path.as_posix(), "image_sha256": _hash(output_dir / image_path),
                                 "mask_sha256": _hash(output_dir / mask_path), "clean_sha256": clean_hash,
                                 "source": Path(source["image"]).relative_to(manifest["root"]).as_posix(),
                                 "source_sha256": source["image_sha256"], "category": source["category"],
                                 "strategy": strategy, "mode": mode, "label": int(strategy != "clean"),
                                 "seed": selected_seed + attempts, "retry_count": attempts,
                                 "anomalous_pixels": int(mask.sum())})
    bank = {"schema_version": 1, "source_split": "validation", "split_digest": manifest["split_digest"],
            "seed": seed, "canonical_image_size": image_size, "source_images": len(records),
            "families": [list(family) for family in FAMILIES], "records": bank_records,
            "storage": "lossless float32 RGB CHW [0,1] images and uint8 binary 1HW masks in NPY files",
            "selection_rule": "maximize harmonic mean of synthetic image AUROC and canonical-mask pixel AP",
            "limitations": "Synthetic ranking is model selection evidence, not a real-defect performance estimate."}
    bank["bank_digest"] = _digest(bank)
    (output_dir / "bank.json").write_text(json.dumps(bank, indent=2) + "\n")
    return bank


class SyntheticBankDataset:
    def __init__(self, bank_path: str | Path, image_size: int | list[int] = 256,
                 verify_content: bool = True):
        self.bank_path = Path(bank_path)
        self.bank = load_bank(bank_path, verify_content=verify_content)
        self.records = self.bank["records"]
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if len(self.image_size) != 2 or min(self.image_size) < 16:
            raise ValueError("Bank model input size must have two dimensions >=16")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        arrays = {name: torch.from_numpy(np.load(self.bank_path.parent / record[name], allow_pickle=False).copy()).float()
                  for name in ("image", "clean", "mask")}
        canonical_mask = arrays["mask"].clone()
        if arrays["image"].shape[-2:] != self.image_size:
            for name in ("image", "clean"):
                arrays[name] = F.interpolate(arrays[name][None], size=self.image_size, mode="bilinear",
                                             align_corners=False, antialias=True)[0]
            arrays["mask"] = F.interpolate(arrays["mask"][None], size=self.image_size, mode="nearest")[0]
        return {**arrays, "canonical_mask": canonical_mask, "label": record["label"],
                "path": record["image"], "strategy": record["strategy"], "mode": record["mode"]}


def harmonic_selection_score(image_auroc: float, pixel_ap: float) -> float:
    if not np.isfinite([image_auroc, pixel_ap]).all() or not 0 <= image_auroc <= 1 or not 0 <= pixel_ap <= 1:
        raise ValueError("Selection metrics must be finite values in [0,1]")
    return float(2 * image_auroc * pixel_ap / (image_auroc + pixel_ap)) if image_auroc + pixel_ap else 0.0


@torch.inference_mode()
def evaluate_bank(model, model_kind: str, bank_path: str | Path, device,
                  image_size: int | list[int], batch_size: int = 8) -> dict:
    """Rank frozen candidates identically; no calibration scores or test labels."""
    dataset = SyntheticBankDataset(bank_path, image_size)
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.eval()
    maps, masks, labels, scores = [], [], [], []
    canonical_size = dataset.bank["canonical_image_size"]
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        images = batch["image"].to(device)
        if model_kind == "patchcore_imagenet":
            activation, image_scores = model.predict_batch(images)
            if activation.ndim == 3:
                activation = activation[:, None]
        else:
            _, second = model(images)
            activation = second if model_kind == "normal_only_denoising_reconstruction" else torch.sigmoid(second)
            flat = activation.flatten(1)
            image_scores = flat.topk(max(1, int(flat.shape[1] * 0.01)), dim=1).values.mean(dim=1)
        scores.extend(image_scores.cpu().tolist())
        canonical_map = F.interpolate(activation, size=(canonical_size, canonical_size),
                                      mode="bilinear", align_corners=False)
        maps.append(canonical_map.cpu().numpy())
        masks.append(batch["canonical_mask"].numpy())
        labels.extend(batch["label"].tolist())
    maps_array = np.concatenate(maps).ravel()
    masks_array = np.concatenate(masks).ravel()
    if not np.isfinite(maps_array).all() or not np.isfinite(scores).all():
        raise ValueError("Nonfinite predictions cannot rank checkpoints")
    image_auroc = float(roc_auc_score(labels, scores))
    pixel_ap = float(average_precision_score(masks_array, maps_array))
    return {"image_auroc": image_auroc, "pixel_average_precision": pixel_ap,
            "selection_score": harmonic_selection_score(image_auroc, pixel_ap),
            "selection_rule": dataset.bank["selection_rule"], "bank_digest": dataset.bank["bank_digest"],
            "split_digest": dataset.bank["split_digest"], "images": len(labels),
            "canonical_image_size": canonical_size, "real_test_images_read": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1_000_042)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    bank = create_bank(args.manifest, args.output, args.seed, args.image_size)
    print(json.dumps({key: value for key, value in bank.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
