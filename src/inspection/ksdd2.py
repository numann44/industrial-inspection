"""Audited real-defect supervision for KolektorSDD2; official test split is preserved."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image

SOURCE = "https://www.vicos.si/resources/kolektorsdd2/"
EXPECTED = {"train": {0: 2085, 1: 246}, "test": {0: 894, 1: 110}}
PREPROCESS = {"mode": "letterbox", "height": 640, "width": 256}


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def split_digest(manifest: dict) -> str:
    fields = ("relative_image", "relative_mask", "image_sha256", "mask_sha256", "label", "group")
    return digest_json({"dataset": "KolektorSDD2", "seed": manifest["seed"],
                        "splits": {s: [{k: r[k] for k in fields} for r in rs]
                                   for s, rs in manifest["splits"].items()}})


def load_manifest(path: str | Path, verify_splits=()) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("digest") != digest_json({k: v for k, v in manifest.items() if k != "digest"}):
        raise ValueError("KSDD2 manifest metadata digest mismatch")
    if manifest.get("split_digest") != split_digest(manifest):
        raise ValueError("KSDD2 portable split digest mismatch")
    for split in verify_splits:
        for record in manifest["splits"][split]:
            for key in ("image", "mask"):
                if hashlib.sha256(Path(record[key]).read_bytes()).hexdigest() != record[key + "_sha256"]:
                    raise ValueError(f"Audited KSDD2 content changed: {record[key]}")
    return manifest


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, i, j):
        self.parents[self.find(j)] = self.find(i)


def _decode_pair(image_path: Path, mask_path: Path, root: Path, original_split: str):
    with Image.open(image_path) as image:
        image.load()
        rgb_image = image.convert("RGB")
        rgb = np.asarray(rgb_image)
        size = image.size
        pixel_hash = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb_image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        tiny = np.asarray(rgb_image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
        dhash = np.packbits((tiny[:, 1:] > tiny[:, :-1]).reshape(-1)).tobytes()
    with Image.open(mask_path) as image:
        image.load()
        array = np.asarray(image)
        if image.size != size or array.ndim != 2 or not set(np.unique(array)).issubset({0, 255}):
            raise ValueError(f"Mask must be aligned binary grayscale: {mask_path}")
        positive = bool(array.any())
        positive_pixels = int(np.count_nonzero(array))
        from .preprocessing import prepare_mask
        resized_positive_pixels = int(prepare_mask(image, PREPROCESS).sum().item())
    record = {"image": str(image_path), "mask": str(mask_path),
              "relative_image": image_path.relative_to(root).as_posix(),
              "relative_mask": mask_path.relative_to(root).as_posix(),
              "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
              "pixel_sha256": pixel_hash, "label": int(positive), "category": "kolektor_surface",
              "defect": "defective" if positive else "good", "original_split": original_split,
              "original_size": list(size), "mask_positive_pixels": positive_pixels,
              "mask_positive_fraction": positive_pixels / (size[0] * size[1]),
              "preprocessed_mask_positive_pixels": resized_positive_pixels}
    return record, small, np.frombuffer(dhash, dtype=np.uint8).copy()


def duplicate_groups(records, thumbnails, hashes):
    """Conservative image similarity groups; not product or acquisition identities."""
    groups = UnionFind(len(records))
    exact, near = [], []
    seen = {}
    for i, record in enumerate(records):
        key = record["pixel_sha256"]
        if key in seen:
            j = seen[key]
            groups.union(j, i)
            exact.append((j, i))
        else:
            seen[key] = i
    lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    hashes = np.stack(hashes)
    for i in range(len(records)):
        if i + 1 == len(records):
            continue
        distances = lookup[np.bitwise_xor(hashes[i], hashes[i + 1:])].sum(axis=1)
        for j in np.flatnonzero(distances <= 2) + i + 1:
            j = int(j)
            if records[i]["pixel_sha256"] == records[j]["pixel_sha256"]:
                continue
            mae = float(np.abs(thumbnails[i] - thumbnails[j]).mean())
            if mae <= 0.004:
                groups.union(i, j)
                near.append((i, j, mae))
    return groups, exact, near


def grouped_stratified_split(records: list[dict], seed: int):
    """Approximately 70/15/15 by class, preserving exact/near image groups."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    group_list = list(grouped.values())
    random.Random(seed).shuffle(group_list)
    group_list.sort(key=len, reverse=True)
    fractions = np.array([0.70, 0.15, 0.15])
    totals = np.array([sum(r["label"] == label for r in records) for label in (0, 1)])
    targets = fractions[:, None] * totals[None]
    counts = np.zeros((3, 2), dtype=float)
    result = {s: [] for s in ("train", "validation", "calibration")}
    names = list(result)
    for group in group_list:
        vector = np.array([sum(r["label"] == label for r in group) for label in (0, 1)])
        costs = [float((((counts[k] + vector - targets[k]) ** 2 -
                         (counts[k] - targets[k]) ** 2) / np.maximum(targets[k], 1)).sum()) for k in range(3)]
        winner = min(range(3), key=lambda k: costs[k])
        result[names[winner]].extend(group)
        counts[winner] += vector
    for name in result:
        result[name].sort(key=lambda r: r["relative_image"])
        if {r["label"] for r in result[name]} != {0, 1}:
            raise ValueError(f"Grouped {name} subset must contain both labels")
    return result


def build_manifest(root: str | Path, seed: int = 42, strict_counts: bool = True,
                   archive_sha256: str | None = None):
    root = Path(root).resolve()
    records, thumbnails, hashes, copied_files = [], [], [], []
    for stem in ("10301", "10301_GT"):
        copy = root / "train" / f"{stem} (copy).png"
        canonical = root / "train" / f"{stem}.png"
        if copy.exists():
            if not canonical.exists() or copy.read_bytes() != canonical.read_bytes():
                raise ValueError(f"Documented copy does not match canonical file: {copy}")
            with Image.open(copy) as image:
                image.load()
            copied_files.append(copy.relative_to(root).as_posix())
    for split in ("train", "test"):
        directory = root / split
        images = sorted(p for p in directory.glob("*.png") if re.fullmatch(r"\d+", p.stem))
        allowed = {p.name for p in images} | {p.stem + "_GT.png" for p in images}
        allowed |= {Path(p).name for p in copied_files if p.startswith(split + "/")}
        extras = {p.name for p in directory.glob("*.png")} - allowed
        if extras:
            raise ValueError(f"Unexpected KSDD2 PNGs: {sorted(extras)}")
        for image in images:
            mask = image.with_name(image.stem + "_GT.png")
            record, thumb, dhash = _decode_pair(image, mask, root, split)
            records.append(record); thumbnails.append(thumb); hashes.append(dhash)
        counts = Counter(r["label"] for r in records if r["original_split"] == split)
        if strict_counts and dict(counts) != EXPECTED[split]:
            raise ValueError(f"Official {split} label counts differ: {dict(counts)}")
    if not records:
        raise ValueError("No KSDD2 canonical images found")
    union, exact, near = duplicate_groups(records, thumbnails, hashes)
    grouped_indices = defaultdict(list)
    for index in range(len(records)):
        grouped_indices[union.find(index)].append(index)
    for indices in grouped_indices.values():
        group_id = min(records[i]["relative_image"] for i in indices)
        for index in indices:
            records[index]["group"] = group_id
    test_groups = {r["group"] for r in records if r["original_split"] == "test"}
    quarantined = [r for r in records if r["original_split"] == "train" and r["group"] in test_groups]
    training = [r for r in records if r["original_split"] == "train" and r["group"] not in test_groups]
    splits = grouped_stratified_split(training, seed)
    splits["test"] = [r for r in records if r["original_split"] == "test"]
    counts = {name: {"total": len(rs), "normal": sum(r["label"] == 0 for r in rs),
                     "defective": sum(r["label"] == 1 for r in rs)} for name, rs in splits.items()}
    manifest = {"schema_version": 1, "dataset": "KolektorSDD2", "root": str(root), "seed": seed,
                "config": {"seed": seed, "validation_fraction": 0.15, "calibration_fraction": 0.15},
                "preprocess": PREPROCESS, "splits": splits, "counts": counts,
                "provenance": {"source": SOURCE, "archive_sha256": archive_sha256,
                               "license": "CC BY-NC-SA 4.0"}}
    manifest["split_digest"] = split_digest(manifest)
    manifest["digest"] = digest_json(manifest)
    pair = lambda i, j: [records[i]["relative_image"], records[j]["relative_image"]]
    audit = {"status": "passed", "dataset": "KolektorSDD2", "source": SOURCE,
             "canonical_images": len(records), "decoded_pngs": len(records) * 2 + len(copied_files),
             "excluded_verified_copy_files": copied_files, "original_counts": {
                 s: dict(Counter("defective" if r["label"] else "normal" for r in records if r["original_split"] == s))
                 for s in ("train", "test")}, "split_counts": counts,
             "exact_pixel_duplicate_pairs": [pair(i, j) for i, j in exact],
             "near_duplicate_pairs": [{"images": pair(i, j), "thumbnail_rgb_mae": mae} for i, j, mae in near],
             "near_duplicate_rule": "64-bit dHash Hamming <=2 and 64x64 bilinear RGB mean absolute difference <=0.004",
             "quarantined_official_train_images": [r["relative_image"] for r in quarantined],
             "image_sizes": dict(sorted(Counter(f"{r['original_size'][0]}x{r['original_size'][1]}" for r in records).items())),
             "split_digest": manifest["split_digest"], "archive_sha256": archive_sha256,
             "preprocess": PREPROCESS, "group_definition": "exact/near pixel similarity only; no supplied product/batch identities",
             "defective_masks_vanishing_after_letterbox": [r["relative_mask"] for r in records
                                                           if r["label"] and not r["preprocessed_mask_positive_pixels"]],
             "limitations": ["Image similarity thresholds cannot establish physical product or acquisition independence.",
                             "Official test membership is preserved; related training images are quarantined instead.",
                             "No .pyb pickle is executed."]}
    return manifest, audit


class KSDD2Dataset:
    def __init__(self, records, preprocess=None):
        self.records = records
        self.preprocess = dict(preprocess or PREPROCESS)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from .preprocessing import prepare_image, prepare_mask
        record = self.records[index]
        with Image.open(record["image"]) as image:
            tensor, valid, _ = prepare_image(image.convert("RGB"), self.preprocess)
        with Image.open(record["mask"]) as mask:
            mask_tensor = prepare_mask(mask.convert("L"), self.preprocess)
        return {"image": tensor, "mask": mask_tensor, "valid_mask": valid,
                "label": record["label"], "path": record["image"],
                "category": record["category"], "defect": record["defect"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ksdd2"))
    parser.add_argument("--output", type=Path, default=Path("data/ksdd2-manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/results/ksdd2-data-verification.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    archive_sha256 = args.archive_sha256
    download_report = args.root.parent / "ksdd2-download-verification.json"
    if archive_sha256 is None and download_report.is_file():
        archive_sha256 = json.loads(download_report.read_text()).get("sha256")
    manifest, audit = build_manifest(args.root, args.seed, archive_sha256=archive_sha256)
    for path, value in ((args.output, manifest), (args.report, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "counts": manifest["counts"],
                      "exact_duplicate_pairs": len(audit["exact_pixel_duplicate_pairs"]),
                      "near_duplicate_pairs": len(audit["near_duplicate_pairs"]),
                      "quarantined": len(audit["quarantined_official_train_images"])}, indent=2))


if __name__ == "__main__":
    main()
