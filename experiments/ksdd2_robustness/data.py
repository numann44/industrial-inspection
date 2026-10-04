"""Native-resolution acquisition transforms and a content-pinned validation bank."""
import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageEnhance

from inspection.checkpointing import config_digest
from inspection.ksdd2 import load_manifest, PREPROCESS
from inspection.preprocessing import prepare_image, prepare_mask

from .protocol import CONDITIONS, immutable_json, sha256


def jpeg(image, quality):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()


def acquisition_transform(image, generator, policy):
    """Return a new PIL image. No resize, geometry change or mask transformation."""
    if float(torch.rand((), generator=generator)) < policy["unchanged_probability"]:
        return image.copy()
    low, high = policy["brightness_range"]
    brightness = low + (high - low) * float(torch.rand((), generator=generator))
    low_quality, high_quality = policy["jpeg_quality_inclusive"]
    quality = int(torch.randint(low_quality, high_quality + 1, (), generator=generator))
    return jpeg(ImageEnhance.Brightness(image).enhance(brightness), quality)


def condition_transform(image, condition):
    if condition == "clean":
        return image.copy()
    if condition == "brightness_0.8":
        return ImageEnhance.Brightness(image).enhance(.8)
    if condition == "brightness_1.2":
        return ImageEnhance.Brightness(image).enhance(1.2)
    if condition == "jpeg_quality_60":
        return jpeg(image, 60)
    raise ValueError("Unknown frozen validation condition")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def verify_records(records, include_masks=True):
    for row in records:
        for key in ("image", "mask") if include_masks else ("image",):
            if sha256(row[key]) != row[f"{key}_sha256"]:
                raise ValueError(f"Audited {key} content changed")


def require_isolated_splits(manifest):
    groups = []
    images = []
    for name in ("train", "validation", "calibration", "test"):
        records = manifest["splits"][name]
        groups.append({r["group"] for r in records})
        images.append({r["image_sha256"] for r in records})
    if any(groups[i] & groups[j] or images[i] & images[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Audited image groups cross split boundaries")
    if any(r["original_split"] != "train" for split in ("train", "validation", "calibration")
           for r in manifest["splits"][split]):
        raise ValueError("Only original training images may enter development")


def prepared_record(record, preprocess, condition="clean"):
    with Image.open(record["image"]) as source:
        image = condition_transform(source.convert("RGB"), condition)
    tensor, valid, _ = prepare_image(image, preprocess)
    with Image.open(record["mask"]) as mask:
        target = prepare_mask(mask.convert("L"), preprocess)
    return {"image": tensor, "mask": target, "valid_mask": valid, "label": record["label"]}


class TrainingDataset:
    def __init__(self, records, cfg):
        if any(r["original_split"] != "train" for r in records):
            raise ValueError("Training dataset accepts only original-training records")
        self.records, self.config = records, cfg
        self.generator = torch.Generator()
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.generator.manual_seed(self.config["training_seed"] + 1_000_000 * epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
            if self.config["candidate"] == "acquisition_aug":
                image = acquisition_transform(image, self.generator, self.config["augmentation"])
            elif self.config["candidate"] != "control":
                raise ValueError("Unknown candidate")
            tensor, valid, _ = prepare_image(image, self.config["preprocess"])
        with Image.open(record["mask"]) as mask:
            target = prepare_mask(mask.convert("L"), self.config["preprocess"])
        return {"image": tensor, "mask": target, "valid_mask": valid, "label": record["label"]}


def create_bank(manifest_path, output_path, preprocess=None):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Validation bank already exists")
    manifest = load_manifest(manifest_path, verify_splits=("validation",))
    require_isolated_splits(manifest)
    preprocess = dict(preprocess or PREPROCESS)
    records = manifest["splits"]["validation"]
    if set(r["label"] for r in records) != {0, 1}:
        raise ValueError("Real validation must contain normal and defective images")
    rows = []
    for group_index, record in enumerate(records):
        for condition in CONDITIONS:
            tensors = prepared_record(record, preprocess, condition)
            rows.append({"group_index": group_index, "condition": condition, "source": record,
                         "tensor_sha256": {key: tensor_hash(tensors[key]) for key in ("image", "mask", "valid_mask")}})
    bank = {"schema_version": "ksdd2-acquisition-bank-1", "source_split": "validation",
            "split_digest": manifest["split_digest"], "conditions": list(CONDITIONS), "preprocess": preprocess,
            "original_images": len(records), "variant_images": len(rows), "records": rows,
            "image_groups": [r["relative_image"] for r in records],
            "correlation": "Four deterministic variants of each original image form one resampling group; not four independent observations.",
            "generation": "Native PIL RGB acquisition transform before letterbox; tensor contents pinned by SHA-256; padding excluded."}
    bank["bank_digest"] = config_digest(bank)
    immutable_json(output_path, bank)
    return bank


def load_bank(path, verify_sources=True):
    bank = json.loads(Path(path).read_text())
    body = {k: v for k, v in bank.items() if k != "bank_digest"}
    if (bank.get("bank_digest") != config_digest(body) or bank.get("source_split") != "validation"
            or bank.get("schema_version") != "ksdd2-acquisition-bank-1"
            or bank["conditions"] != list(CONDITIONS)):
        raise ValueError("Frozen validation bank identity differs")
    if verify_sources:
        sources = {row["source"]["image"]: row["source"] for row in bank["records"]}
        verify_records(list(sources.values()))
    return bank


def verify_bank_membership(bank, manifest):
    expected = manifest["splits"]["validation"]
    if bank["split_digest"] != manifest["split_digest"] or bank["original_images"] != len(expected):
        raise ValueError("Validation bank does not match the original validation split")
    if len(bank["records"]) != len(expected) * len(CONDITIONS):
        raise ValueError("Every validation source must have exactly four conditions")
    expected_rows = [(i, condition, row) for i, row in enumerate(expected) for condition in CONDITIONS]
    for actual, (index, condition, record) in zip(bank["records"], expected_rows):
        if actual["group_index"] != index or actual["condition"] != condition or actual["source"] != record:
            raise ValueError("Bank contains a substituted source or condition")


class ValidationDataset:
    def __init__(self, bank_path):
        self.bank = load_bank(bank_path, verify_sources=False)
        self.records = self.bank["records"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        tensors = prepared_record(row["source"], self.bank["preprocess"], row["condition"])
        if any(tensor_hash(tensors[key]) != row["tensor_sha256"][key] for key in ("image", "mask", "valid_mask")):
            raise ValueError("Validation tensor contents differ from the frozen bank")
        return {**tensors, "group_index": row["group_index"], "condition_index": CONDITIONS.index(row["condition"])}
