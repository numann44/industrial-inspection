"""Inspect one supplied photograph using a frozen model and calibration threshold."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inspection.model import InspectionModel, anomaly_score
from inspection.train import select_device


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _model_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def predict(checkpoint_path, image_path, output, device_name="auto"):
    checkpoint_path, image_path, output = map(Path, (checkpoint_path, image_path, output))
    targets = (output / "prediction.json", output / "heatmap.png", output / "overlay.png")
    if any(target.exists() for target in targets):
        raise FileExistsError("Prediction artifacts already exist; choose a new output directory")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("Checkpoint threshold must be finite")
    config = checkpoint["config"]
    category = config["category"]
    image_size = int(checkpoint["image_size"])
    if image_size < 16:
        raise ValueError("Checkpoint image_size must be at least 16")
    device = select_device(device_name)
    model = InspectionModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # This matches InspectionDataset exactly: RGB, square bilinear resize, [0,1].
    with Image.open(image_path) as image:
        original = image.convert("RGB")
        array = np.asarray(original.resize((image_size, image_size), Image.Resampling.BILINEAR),
                           dtype=np.float32).copy() / 255
    tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
    with torch.inference_mode():
        for _ in range(3):
            model(tensor)
        _synchronize(device)
        started = time.perf_counter()
        _, logits = model(tensor)
        score_tensor = anomaly_score(logits)
        _synchronize(device)
        inference_ms = (time.perf_counter() - started) * 1000
        score = float(score_tensor.cpu()[0])
        heatmap = torch.sigmoid(logits).cpu().numpy()[0, 0]
    if not np.isfinite(score) or not np.isfinite(heatmap).all():
        raise RuntimeError("Non-finite model prediction")
    defective = score >= threshold
    quantized = np.clip(np.round(heatmap * 255), 0, 255).astype(np.uint8)
    # The PNG stores uncalibrated pixel activations at the model resolution.
    heatmap_image = Image.fromarray(quantized)
    enlarged = np.asarray(Image.fromarray(heatmap).resize(
        original.size, Image.Resampling.BILINEAR), dtype=np.float32)
    rgb = np.asarray(original, dtype=np.float32)
    color = np.zeros_like(rgb)
    color[:, :, 0], color[:, :, 1] = 255, 48
    alpha = np.clip(enlarged, 0, 1)[:, :, None] * 0.65
    overlay = Image.fromarray(np.clip(rgb * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8))
    result = {
        "image": str(image_path.resolve()), "category": category,
        "category_source": "checkpoint training configuration; no automatic category recognition",
        "score": score, "score_definition": "mean of highest 1% sigmoid pixel activations; not a calibrated defect probability",
        "threshold": threshold, "decision_rule": "score >= threshold",
        "decision": "defective" if defective else "good", "predicted_defective": defective,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "model_sha256": _model_digest(checkpoint["model_state"]),
        "model_image_size": image_size, "original_image_size": list(original.size),
        "device": str(device), "warmup_forwards": 3, "inference_ms": inference_ms,
        "inference_scope": "warm model forward and image scoring; excludes loading, preprocessing, transfer and artifact writing",
        "heatmap": "heatmap.png", "overlay": "overlay.png",
        "heatmap_encoding": "8-bit grayscale sigmoid activations at model resolution; no calibrated pixel decision threshold",
        "smoke_checkpoint": bool(checkpoint.get("smoke_run", False)),
        "limitations": [
            "The supplied photograph must match the checkpoint category and training imaging conditions; other inputs may be unreliable.",
            "Synthetic-defect training and a small normal calibration set do not guarantee accuracy on new parts.",
            "Overlay colors visualize uncalibrated activations, not defect probabilities or confirmed defect boundaries.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    heatmap_image.save(targets[1])
    overlay.save(targets[2])
    targets[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image, args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
