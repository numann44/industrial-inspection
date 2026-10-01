"""Normal-only denoising reconstruction baseline with independent calibration."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import InspectionDataset, load_manifest
from .evaluate import compute_metrics, original_pixel_metrics
from .model import SmallUNet
from .train import select_device

MODEL_KIND = "normal_only_denoising_reconstruction"
SCORE_DEFINITION = "mean top 1% of per-pixel channel-mean absolute reconstruction error; no sigmoid or probability calibration"


class ReconstructionBaseline(nn.Module):
    """Random-initialized U-Net; second output is raw reconstruction error."""
    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.network = SmallUNet(3, 3, base_channels)

    def forward(self, image):
        reconstruction = torch.sigmoid(self.network(image))
        error = (image - reconstruction).abs().mean(dim=1, keepdim=True)
        return reconstruction, error


def reconstruction_score(error_map: torch.Tensor) -> torch.Tensor:
    pixels = error_map.flatten(1)
    count = max(1, int(pixels.shape[1] * 0.01))
    return pixels.topk(count, dim=1).values.mean(dim=1)


def threshold_from_normal_scores(scores, quantile: float = 0.95) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < quantile < 1:
        raise ValueError("Finite, nonempty normal calibration scores and a quantile in (0,1) are required")
    return float(np.quantile(scores, quantile, method="linear"))


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _write_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _normal_epoch(model, loader, device, optimizer=None, noise_std=0.0):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            clean = batch["image"].to(device)
            noisy = (clean + torch.randn_like(clean) * noise_std).clamp(0, 1) if optimizer is not None and noise_std else clean
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(noisy)
            loss = nn.functional.l1_loss(reconstruction, clean)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite reconstruction loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * len(clean)
            count += len(clean)
    if not count:
        raise ValueError("Normal data split is empty")
    return total / count


@torch.inference_mode()
def calibrate_baseline(model, records, image_size, batch_size, device, quantile):
    model.eval()
    loader = DataLoader(InspectionDataset(records, image_size, include_masks=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    for batch in loader:
        _, errors = model(batch["image"].to(device))
        scores.extend(reconstruction_score(errors).cpu().tolist())
    return threshold_from_normal_scores(scores, quantile), scores


def train_baseline(manifest_path, config_path, output, epochs=None, device_override=None):
    config = json.loads(Path(config_path).read_text())
    if epochs is not None:
        config["epochs"] = epochs
    if device_override is not None:
        config["device"] = device_override
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["base_channels"] < 1 or config["image_size"] < 16:
        raise ValueError("Positive epochs/batch_size/base_channels and image_size >=16 required")
    if config["learning_rate"] <= 0 or not 0 <= config["noise_std"] <= 0.2 or not 0 < config["threshold_quantile"] < 1:
        raise ValueError("Invalid learning rate, noise standard deviation, or threshold quantile")
    manifest = load_manifest(manifest_path, verify_splits=("train", "validation", "calibration"))
    splits = manifest["splits"]
    for name in ("train", "validation", "calibration"):
        if not splits[name] or any(record["label"] != 0 or record["category"] != config["category"] for record in splits[name]):
            raise ValueError(f"{name} must be nonempty normal-only data for the configured category")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists():
        raise FileExistsError("A checkpoint already exists; choose a distinct baseline output directory")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(config["device"])
    model_config = {"base_channels": config["base_channels"]}
    model = ReconstructionBaseline(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    size, batch_size = config["image_size"], config["batch_size"]
    loaders = {name: DataLoader(InspectionDataset(splits[name], size, include_masks=False),
                                batch_size=batch_size, shuffle=name == "train", num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
               for name in ("train", "validation")}
    history, best_state, best_epoch = [], None, None
    best_loss = float("inf")
    started = time.perf_counter()
    _write_json(output / "config.json", config)
    print(json.dumps({"model_kind": MODEL_KIND, "device": str(device),
                      "parameters": sum(parameter.numel() for parameter in model.parameters()),
                      "split_counts": {name: len(splits[name]) for name in ("train", "validation", "calibration")}}), flush=True)
    for epoch in range(config["epochs"]):
        _sync(device)
        epoch_start = time.perf_counter()
        training_loss = _normal_epoch(model, loaders["train"], device, optimizer, config["noise_std"])
        validation_loss = _normal_epoch(model, loaders["validation"], device)
        _sync(device)
        row = {"epoch": epoch + 1, "train_l1": training_loss, "validation_l1": validation_loss,
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        _write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    threshold, scores = calibrate_baseline(model, splits["calibration"], size, batch_size,
                                          device, config["threshold_quantile"])
    checkpoint = {"schema_version": 1, "model_kind": MODEL_KIND, "model_state": best_state,
                  "model_config": model_config, "config": config, "image_size": size,
                  "threshold": threshold, "threshold_source": "separate normal calibration split",
                  "score_definition": SCORE_DEFINITION, "manifest_digest": manifest["digest"],
                  "split_digest": manifest["split_digest"], "history": history, "best_epoch": best_epoch,
                  "calibration": {"count": len(scores), "scores": scores, "quantile": config["threshold_quantile"],
                                  "quantile_method": "linear", "calibration_exceedances": sum(score >= threshold for score in scores)},
                  "elapsed_seconds": time.perf_counter() - started, "device": str(device),
                  "torch_version": str(torch.__version__),
                  "learning_target": "clean normal image; optional Gaussian noise is added only to training inputs",
                  "model_selection": "minimum clean normal validation L1; no test-image reads during training"}
    temporary = output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output / "checkpoint.pt")
    _write_json(output / "summary.json", {key: value for key, value in checkpoint.items() if key not in ("model_state", "history")})
    print(json.dumps({"checkpoint": str(output / "checkpoint.pt"), "best_epoch": best_epoch,
                      "threshold": threshold, "elapsed_seconds": checkpoint["elapsed_seconds"]}), flush=True)
    return checkpoint


def evaluate_baseline(manifest_path, checkpoint_path, device_name="auto", batch_size=8, pixel_space="original"):
    manifest = load_manifest(manifest_path, verify_splits=("test",))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != MODEL_KIND or checkpoint.get("split_digest") != manifest["split_digest"]:
        raise ValueError("Expected a reconstruction baseline checkpoint with the same portable split digest")
    if "threshold" not in checkpoint or batch_size < 1 or pixel_space not in ("original", "model"):
        raise ValueError("Checkpoint threshold, positive batch size and valid pixel space are required")
    device = select_device(device_name)
    model = ReconstructionBaseline(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    records = manifest["splits"]["test"]
    dataset = InspectionDataset(records, checkpoint["image_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores, maps, masks, labels, timings = [], [], [], [], []
    with torch.inference_mode():
        warmup = dataset[0]["image"][None].to(device)
        for _ in range(3):
            model(warmup)
        _sync(device)
        for batch in loader:
            images = batch["image"].to(device)
            _, errors = model(images)
            scores.extend(reconstruction_score(errors).cpu().tolist())
            maps.append(errors.cpu().numpy())
            masks.append(batch["mask"].numpy())
            labels.extend(batch["label"].tolist())
        for index in range(min(32, len(dataset))):
            image = dataset[index]["image"][None].to(device)
            _sync(device)
            start = time.perf_counter()
            _, errors = model(image)
            reconstruction_score(errors)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000)
    maps = np.concatenate(maps)
    masks = np.concatenate(masks)
    threshold = float(checkpoint["threshold"])
    metrics = compute_metrics(labels, scores, masks, maps, threshold, include_pixel_metric=pixel_space == "model")
    if pixel_space == "original":
        metrics.update(original_pixel_metrics(records, maps))
        metrics["evaluation_resolution"] = "original resolution per image"
    else:
        metrics.update({"pixel_metric_space": "model", "pixel_aggregation": "pooled pixels at model resolution"})
    metrics["recall_by_defect"] = {}
    for record, score in zip(records, scores):
        if record["label"]:
            key = f"{record['category']}/{record['defect']}"
            group = metrics["recall_by_defect"].setdefault(key, {"images": 0, "detected": 0})
            group["images"] += 1
            group["detected"] += int(score >= threshold)
    for group in metrics["recall_by_defect"].values():
        group["recall"] = group["detected"] / group["images"]
    metrics.update({"model_kind": MODEL_KIND, "manifest_digest": manifest["digest"],
                    "split_digest": manifest["split_digest"], "checkpoint": str(Path(checkpoint_path).resolve()),
                    "threshold_source": checkpoint["threshold_source"], "score_definition": SCORE_DEFINITION,
                    "latency": {"single_image_median_ms": float(np.median(timings)),
                                "single_image_p95_ms": float(np.quantile(timings, 0.95)),
                                "single_image_measurements": len(timings), "warmup_forwards": 3,
                                "device": str(device), "scope": "forward and image scoring; excludes loading and device transfer"},
                    "predictions": [{"image": record["image"], "category": record["category"],
                                     "defect": record["defect"], "label": record["label"], "score": float(score),
                                     "predicted_defective": bool(score >= threshold)} for record, score in zip(records, scores)],
                    "limitations": ["Skip connections can reconstruct abnormal pixels and weaken anomaly separation.",
                                     "Small normal calibration sets do not guarantee future false alarm rates.",
                                     "Native-mask pixel AP uses upsampled predictions; input resizing can erase small defects."]})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    training = modes.add_parser("train")
    training.add_argument("--manifest", required=True, type=Path)
    training.add_argument("--config", type=Path, default=Path("configs/metal_nut_reconstruction.json"))
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--epochs", type=int)
    training.add_argument("--device", default=None, choices=["auto", "cpu", "mps", "cuda"])
    evaluation = modes.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True, type=Path)
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    evaluation.add_argument("--pixel-space", default="original", choices=["original", "model"])
    args = parser.parse_args()
    if args.mode == "train":
        train_baseline(args.manifest, args.config, args.output, args.epochs, args.device)
    else:
        metrics = evaluate_baseline(args.manifest, args.checkpoint, args.device, args.batch_size, args.pixel_space)
        _write_json(args.output, metrics)
        print(json.dumps({key: value for key, value in metrics.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
