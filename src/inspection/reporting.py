"""Image-level uncertainty, native-mask error groups and portable evidence."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score


def checkpoint_provenance(checkpoint_path: str | Path) -> dict:
    path = Path(checkpoint_path)
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    source_hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                     for source in sorted(Path(__file__).parent.glob("*.py"))}
    return {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evaluation_code_revision": revision, "evaluation_worktree_dirty": dirty,
            "evaluation_source_sha256": source_hashes}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials == 0:
        return None
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("Binomial counts must satisfy 0 <= successes <= trials")
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def image_uncertainty(labels, scores, threshold: float, seed: int = 42, bootstrap_samples: int = 1000) -> dict:
    """Stratified image bootstrap and Wilson rates; pixels are not independent."""
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or not len(labels) or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Finite scores and aligned nonempty binary labels required")
    if bootstrap_samples < 1 or not np.isfinite(threshold):
        raise ValueError("Bootstrap count must be positive and threshold finite")
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    predictions = scores >= threshold
    tp, fp = int(predictions[positive].sum()), int(predictions[negative].sum())
    auc_interval = None
    if len(positive) and len(negative):
        random = np.random.default_rng(seed)
        values = []
        for _ in range(bootstrap_samples):
            indices = np.concatenate((random.choice(positive, len(positive), replace=True),
                                      random.choice(negative, len(negative), replace=True)))
            values.append(roc_auc_score(labels[indices], scores[indices]))
        auc_interval = np.quantile(values, [0.025, 0.975]).tolist()
    return {"confidence_level": 0.95, "image_auroc_interval": auc_interval,
            "auroc_method": "percentile stratified bootstrap of independent test images",
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "defect_recall_interval": wilson_interval(tp, len(positive)),
            "normal_false_alarm_rate_interval": wilson_interval(fp, len(negative)),
            "precision_interval": wilson_interval(tp, tp + fp), "rate_method": "Wilson score interval",
            "defective_images": len(positive), "normal_images": len(negative),
            "limitations": "Intervals describe finite-image sampling uncertainty, not training-seed variability, domain shift or independent-pixel uncertainty."}


def defect_area_groups(records: list[dict], scores, threshold: float) -> dict:
    """Predeclared native-mask area bins, evaluated at the frozen image threshold."""
    names = ("tiny_below_0.1_percent", "small_0.1_to_1_percent", "medium_1_to_5_percent", "large_at_least_5_percent")
    groups = {name: {"images": 0, "detected": 0, "missed": 0} for name in names}
    for record, score in zip(records, scores):
        if not record["label"]:
            continue
        with Image.open(record["mask"]) as mask:
            fraction = float((np.asarray(mask) > 0).mean())
        index = 0 if fraction < 0.001 else 1 if fraction < 0.01 else 2 if fraction < 0.05 else 3
        group = groups[names[index]]
        group["images"] += 1
        group["detected"] += int(score >= threshold)
        group["missed"] += int(score < threshold)
    for group in groups.values():
        group["recall"] = group["detected"] / group["images"] if group["images"] else None
        group["recall_interval"] = wilson_interval(group["detected"], group["images"])
    return {"area_definition": "fraction of original image pixels with nonzero ground-truth mask",
            "boundaries": [0.001, 0.01, 0.05], "groups": groups}


def select_gallery_cases(records, scores, threshold, limit_per_group=3) -> dict[str, list[int]]:
    if limit_per_group < 1:
        raise ValueError("Gallery limit must be positive")
    groups = {name: [] for name in ("FP", "FN", "TP", "TN")}
    for index, (record, score) in enumerate(zip(records, scores)):
        group = "TP" if score >= threshold and record["label"] else "FP" if score >= threshold else "FN" if record["label"] else "TN"
        groups[group].append(index)
    groups["FP"].sort(key=lambda index: (-scores[index], records[index]["image"]))
    groups["FN"].sort(key=lambda index: (scores[index], records[index]["image"]))
    # Successful examples cover score positions, rather than only the easiest wins.
    for name in ("TP", "TN"):
        ordered = sorted(groups[name], key=lambda index: (scores[index], records[index]["image"]))
        positions = np.linspace(0, len(ordered) - 1, min(limit_per_group, len(ordered)), dtype=int) if ordered else []
        groups[name] = [ordered[position] for position in positions]
    return {name: indices[:limit_per_group] for name, indices in groups.items()}


def write_failure_gallery(records, native_maps, scores, threshold, output: str | Path,
                          limit_per_group: int = 3, display_max: float | None = None) -> dict:
    """Separate original/prediction/ground truth, all at original dimensions."""
    output = Path(output)
    kolektor = bool(records) and all(record["category"] == "kolektor_surface" for record in records)
    dataset = "KolektorSDD2" if kolektor else "MVTec AD"
    source_url = "https://www.vicos.si/resources/kolektorsdd2/" if kolektor else "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
    credit = "ViCoS, University of Ljubljana (KolektorSDD2)" if kolektor else "MVTec Software GmbH"
    display_max = max(2 * threshold, 1e-8) if display_max is None else float(display_max)
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError("Gallery display maximum must be finite and positive")
    output.mkdir(parents=True, exist_ok=True)
    selected = select_gallery_cases(records, scores, threshold, limit_per_group)
    items = []
    for group, indices in selected.items():
        for index in indices:
            record = records[index]
            with Image.open(record["image"]) as image:
                original = image.convert("RGB")
            rgb = np.asarray(original, dtype=np.float32)
            activation = np.asarray(native_maps[index], dtype=np.float32).squeeze()
            if activation.shape != rgb.shape[:2]:
                raise ValueError("Gallery maps must already be restored to original image dimensions")
            red = np.zeros_like(rgb)
            red[:, :, 0], red[:, :, 1] = 255, 48
            alpha = np.clip(activation / display_max, 0, 1)[:, :, None] * 0.65
            prediction = (rgb * (1 - alpha) + red * alpha).astype(np.uint8)
            truth = rgb.copy()
            if record["mask"]:
                with Image.open(record["mask"]) as image:
                    mask = np.asarray(image) > 0
                green = np.zeros_like(rgb)
                green[:, :, 1] = 255
                truth = np.where(mask[:, :, None], rgb * 0.45 + green * 0.55, rgb)
            width, height = original.size
            header_height = 64
            font = ImageFont.load_default(size=max(12, min(24, width // 28)))
            canvas = Image.new("RGB", (width * 3, height + header_height), "white")
            canvas.paste(original, (0, header_height))
            canvas.paste(Image.fromarray(prediction), (width, header_height))
            canvas.paste(Image.fromarray(truth.astype(np.uint8)), (width * 2, header_height))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"{group} | {record['category']}/{record['defect']} | score={scores[index]:.6f} threshold={threshold:.6f}", fill="black", font=font)
            for position, title in enumerate(("Original", f"Activation: fixed display 0..{display_max:.6g}", "Ground truth annotation")):
                draw.text((position * width + 8, 35), title, fill="black", font=font)
            name = f"{group}_{index:04d}_{record['category']}_{record['defect']}.png"
            canvas.save(output / name)
            items.append({"file": name, "group": group, "category": record["category"],
                          "defect": record["defect"], "source": record.get("relative_image", f"{record['category']}/test/{record['defect']}/{Path(record['image']).name}"),
                          "label": record["label"], "score": float(scores[index]),
                          "predicted_defective": bool(scores[index] >= threshold)})
    result = {"selection": "highest-scoring FP; lowest-scoring FN; evenly spaced successful-case score ranks",
              "groups": {group: len(indices) for group, indices in selected.items()}, "items": items,
              "threshold": float(threshold), "display_max": display_max,
              "display_rule": "fixed per-checkpoint range, never per-image min/max; saturation is not a pixel decision",
              "dataset": dataset, "source_url": source_url,
              "license": f"{dataset} derivatives: CC BY-NC-SA 4.0"}
    (output / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "ATTRIBUTION.md").write_text(f"Images are derived from [{dataset}]({source_url}), "
                                            f"{credit}, under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
                                            "Changes: model activation overlays, ground-truth overlays, panel composition and labels. "
                                            "These derivative images retain that license.\n")
    return result
