"""Render native failure cases at a shared 0..1 activation scale after evaluation."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from inspection.checkpointing import atomic_json, config_digest
from inspection.engine import InspectionEngine
from inspection.reporting import select_gallery_cases
from experiments.ksdd2_robustness.protocol import sha256
from scripts.evaluate_ksdd2_robustness import passed_inputs, validate_cached
from scripts.run_study import scheduler_lock


def render(root):
    out = root / "outputs/ksdd2-robustness-evaluation"
    torch.set_num_threads(1)
    with scheduler_lock(root / "outputs/mps-study.lock"):
        _, _, assets, manifest = passed_inputs(root)
        if json.loads((out / "status.json").read_text())["state"] != "complete":
            raise ValueError("Complete numeric comparison before rendering error cases")
        freeze = json.loads((out / "pre-evaluation-freeze.json").read_text())
        for asset in assets:
            evaluation = validate_cached(json.loads((out / asset["candidate"] / "evaluation.json").read_text()),
                                         asset, manifest["split_digest"], "evaluation", config_digest(freeze))
            records = manifest["splits"]["test"]
            if [r["image"] for r in records] != [r["image"] for r in evaluation["predictions"]]:
                raise ValueError("Gallery prediction/source ordering differs")
            scores = [r["score"] for r in evaluation["predictions"]]
            selected = select_gallery_cases(records, scores, asset["threshold"], limit_per_group=3)
            directory = root / "docs/results/ksdd2-robustness-galleries" / asset["candidate"]
            directory.mkdir(parents=True, exist_ok=True)
            engine = InspectionEngine(asset["checkpoint"], device="mps")
            items = []
            for group, indices in selected.items():
                for index in indices:
                    record = records[index]
                    inspected = engine.inspect(record["image"])
                    original = inspected["image"]
                    rgb = np.asarray(original, dtype=np.float32)
                    activation = inspected["native_map"]
                    with Image.open(record["mask"]) as mask:
                        target = np.asarray(mask) > 0
                    if activation.shape != target.shape or activation.shape != rgb.shape[:2]:
                        raise ValueError("Gallery requires native map and original mask alignment")
                    red, green = np.zeros_like(rgb), np.zeros_like(rgb)
                    red[:, :, 0], red[:, :, 1], green[:, :, 1] = 255, 48, 255
                    alpha = np.clip(activation, 0, 1)[:, :, None] * .65
                    prediction = (rgb * (1 - alpha) + red * alpha).astype(np.uint8)
                    truth = np.where(target[:, :, None], rgb * .45 + green * .55, rgb).astype(np.uint8)
                    width, height = original.size
                    panel = Image.new("RGB", (width * 3, height + 96), "white")
                    for offset, image in enumerate((original, Image.fromarray(prediction), Image.fromarray(truth))):
                        panel.paste(image, (offset * width, 96))
                    draw = ImageDraw.Draw(panel); font = ImageFont.load_default(size=13)
                    draw.text((8, 6), f"{asset['candidate']} | {group} | {record['relative_image']} | exploratory", fill="black", font=font)
                    draw.text((8, 28), f"Evaluation score={scores[index]:.6g}   frozen threshold={asset['threshold']:.6g}", fill="black", font=font)
                    draw.text((8, 50), "Shared display 0..1; activation is not a defect probability or binary pixel mask", fill="black", font=font)
                    for offset, text in enumerate(("Original", "Model activation", "Ground truth")):
                        draw.text((offset * width + 8, 74), text, fill="black", font=font)
                    name = f"{group}_{index:04d}.png"
                    panel.save(directory / name)
                    items.append({"file": name, "sha256": sha256(directory / name), "group": group,
                                  "source": record["relative_image"], "source_sha256": record["image_sha256"],
                                  "score": scores[index], "threshold": asset["threshold"], "label": record["label"],
                                  "map_single_image_score": inspected["summary"]["score"]})
            atomic_json(directory / "index.json", {"candidate": asset["candidate"], "checkpoint_sha256": asset["checkpoint_sha256"],
                "experimental_status": "exploratory", "evaluation_freeze_digest": config_digest(freeze),
                "renderer_sha256": sha256(Path(__file__)), "display_range": [0, 1],
                "selection": "Highest-scoring FP; lowest-scoring FN; evenly spaced TP/TN score ranks; up to3 each",
                "map_scope": "Same fixed weights, single-image native map; captions/decisions use frozen batched evaluation scores",
                "groups": {k: len(v) for k,v in selected.items()}, "items": items})
            (directory / "ATTRIBUTION.md").write_text(
                "Images and original masks: [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/), Kolektor Group; "
                "Jakob Božič, Domen Tabernik and Danijel Skočaj, Computers in Industry (2021). "
                "[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
                "Changes: model activation overlays, ground-truth overlays, panel composition and scientific-notation labels. "
                "These derivative figures retain the dataset license and are used for noncommercial research. "
                "They are real evaluated examples, not generated model results.\n")


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1])
