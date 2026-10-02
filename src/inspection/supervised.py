"""From-scratch real-defect segmentation, explicitly separate from normal-only training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .checkpointing import (atomic_json, atomic_torch_save, capture_rng, collect_provenance,
                            config_digest, cpu_copy, prepare_output, restore_rng, validate_resume)
from .ksdd2 import KSDD2Dataset, load_manifest, PREPROCESS
from .model import SegmentationOnlyModel
from .train import select_device

MODEL_KIND = "supervised_segmentation"
SCORE_DEFINITION = "mean highest 1% sigmoid segmentation pixels within valid letterbox area; not a calibrated probability"


def _supervised_sources():
    directory = Path(__file__).resolve().parent
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("supervised.py", "ksdd2.py")}


class BalancedBatchSampler(Sampler):
    """Each training batch has equal normal/defective examples, with replacement."""
    def __init__(self, records, batch_size: int, generator: torch.Generator):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Balanced batches require a positive even batch size >=2")
        self.indices = [torch.tensor([i for i, r in enumerate(records) if r["label"] == label])
                        for label in (0, 1)]
        if any(not len(indices) for indices in self.indices):
            raise ValueError("Balanced batches require both labels")
        self.batch_size, self.generator = batch_size, generator
        self.batches = math.ceil(len(records) / batch_size)

    def __len__(self):
        return self.batches

    def __iter__(self):
        for _ in range(self.batches):
            batch = torch.cat([indices[torch.randint(len(indices), (self.batch_size // 2,),
                                                    generator=self.generator)] for indices in self.indices])
            yield batch[torch.randperm(len(batch), generator=self.generator)].tolist()


def segmentation_loss(logits, target, valid, positive_pixel_weight=3.0):
    valid = valid.to(logits.dtype)
    target = target.to(logits.dtype)
    if logits.shape != target.shape or target.shape != valid.shape or not valid.any():
        raise ValueError("Aligned logits, masks and nonempty valid pixels are required")
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none",
                                                   pos_weight=logits.new_tensor(positive_pixel_weight))
    bce = (pixel_loss * valid).sum() / valid.sum()
    probabilities = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice, {"bce": float(bce.detach().cpu()), "dice": float(dice.detach().cpu())}


def valid_anomaly_score(logits, valid):
    probabilities = torch.sigmoid(logits).flatten(1)
    validity = valid.bool().flatten(1)
    if probabilities.shape != validity.shape or not validity.any(dim=1).all():
        raise ValueError("Every image requires matching valid pixels")
    return torch.stack([values[selected].topk(max(1, int(selected.sum().item() * 0.01))).values.mean()
                        for values, selected in zip(probabilities, validity)])


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def validation_metrics(model, loader, device, positive_pixel_weight=3.0):
    model.eval()
    scores, labels, pixel_scores, pixel_targets = [], [], [], []
    total_loss, count = 0.0, 0
    for batch in loader:
        image = batch["image"].to(device)
        target, valid = [batch[key].to(device) for key in ("mask", "valid_mask")]
        _, logits = model(image)
        loss, _ = segmentation_loss(logits, target, valid, positive_pixel_weight)
        scores.extend(valid_anomaly_score(logits, valid).cpu().tolist())
        labels.extend(batch["label"].tolist())
        validity = batch["valid_mask"].numpy().astype(bool)
        pixel_scores.append(torch.sigmoid(logits).cpu().numpy()[validity])
        pixel_targets.append(batch["mask"].numpy()[validity].astype(bool))
        total_loss += float(loss.cpu()) * len(image)
        count += len(image)
    if not count or len(set(labels)) != 2:
        raise ValueError("Supervised validation must contain both labels")
    image_auroc = float(roc_auc_score(labels, scores))
    pixel_ap = float(average_precision_score(np.concatenate(pixel_targets), np.concatenate(pixel_scores)))
    return {"loss": total_loss / count, "image_auroc": image_auroc,
            "pixel_average_precision": pixel_ap,
            "selection_harmonic_mean": 2 * image_auroc * pixel_ap / max(image_auroc + pixel_ap, 1e-12),
            "pixel_metric_space": "letterboxed model resolution, padding excluded; final evaluation uses original masks"}


@torch.inference_mode()
def calibrate(model, records, preprocess, batch_size, device, quantile):
    if not records or any(r["label"] != 0 for r in records):
        raise ValueError("Calibration accepts only separate normal records")
    model.eval()
    loader = DataLoader(KSDD2Dataset(records, preprocess), batch_size=batch_size, num_workers=0)
    scores = []
    for batch in loader:
        _, logits = model(batch["image"].to(device))
        scores.extend(valid_anomaly_score(logits, batch["valid_mask"].to(device)).cpu().tolist())
    threshold = float(np.quantile(scores, quantile, method="linear"))
    return {"split": "calibration", "normal_count": len(scores), "scores": scores,
            "quantile": quantile, "quantile_method": "linear", "threshold": threshold,
            "decision_rule": ">=", "calibration_exceedances": sum(score >= threshold for score in scores),
            "warning": "An empirical quantile does not guarantee future false-alarm rates."}


def train(manifest_path, config_path, output, resume=None, stop_after_epoch=None, device_override=None, seed_override=None):
    config = json.loads(Path(config_path).read_text())
    if device_override is not None:
        config["device"] = device_override
    if seed_override is not None:
        config["training_seed"] = seed_override
    config.setdefault("preprocess", dict(PREPROCESS))
    config.setdefault("training_seed", 42)
    config.setdefault("threshold_quantile", 0.90)
    config.setdefault("patience", 15)
    config.setdefault("positive_pixel_weight", 3.0)
    if not 1 <= config["epochs"] <= 100 or config["patience"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Use 1–100 epochs, positive patience and learning rate")
    if not 0 < config["threshold_quantile"] < 1 or config["positive_pixel_weight"] <= 0:
        raise ValueError("Invalid threshold quantile or positive pixel loss weight")
    seed = config["training_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = select_device(config.get("device", "auto"))
    manifest = load_manifest(manifest_path, verify_splits=("train", "validation"))
    output = prepare_output(output, resume)
    model_config = {"base_channels": config["base_channels"]}
    model = SegmentationOnlyModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_records, validation_records = [manifest["splits"][s] for s in ("train", "validation")]
    train_loader = DataLoader(KSDD2Dataset(train_records, config["preprocess"]),
                              batch_sampler=BalancedBatchSampler(train_records, config["batch_size"], generator),
                              num_workers=0)
    validation_loader = DataLoader(KSDD2Dataset(validation_records, config["preprocess"]),
                                   batch_size=config["batch_size"], num_workers=0, shuffle=False)
    provenance = collect_provenance(config, manifest)
    provenance["supervised_source_files"] = _supervised_sources()
    history, best_state, best_metric, best_epoch, first_epoch, stale_epochs = [], None, -1.0, None, 0, 0
    if resume:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        validate_resume(previous, config, manifest["split_digest"], MODEL_KIND, actual_device=device)
        if previous["provenance"].get("supervised_source_files") != _supervised_sources():
            raise ValueError("Supervised training/data source changed since checkpoint")
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        restore_rng(previous["rng_state"], generator)
        history, best_state = previous["history"], previous["best_model_state"]
        best_metric, best_epoch = previous["best_metric"], previous["best_epoch"]
        first_epoch, stale_epochs = previous["epoch"], previous["stale_epochs"]
        provenance = previous["provenance"]
    atomic_json(output / "config.json", config)
    started = time.perf_counter()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    stopped_early = False
    for epoch in range(first_epoch, config["epochs"]):
        if stale_epochs >= config["patience"]:
            stopped_early = True
            break
        model.train()
        _sync(device)
        epoch_start = time.perf_counter()
        total_loss, count = 0.0, 0
        for batch in train_loader:
            image, target, valid = [batch[key].to(device) for key in ("image", "mask", "valid_mask")]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(image)
            loss, _ = segmentation_loss(logits, target, valid, config["positive_pixel_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised training loss")
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(image)
            count += len(image)
        validation = validation_metrics(model, validation_loader, device, config["positive_pixel_weight"])
        metric = validation["selection_harmonic_mean"]
        if metric > best_metric:
            best_metric, best_epoch, best_state, stale_epochs = metric, epoch + 1, cpu_copy(model.state_dict()), 0
        else:
            stale_epochs += 1
        _sync(device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / count, "validation": validation,
               "seconds": time.perf_counter() - epoch_start,
               "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
        history.append(row)
        last = {"schema_version": 2, "resumable": True, "model_kind": MODEL_KIND,
                "model_state": cpu_copy(model.state_dict()), "optimizer_state": cpu_copy(optimizer.state_dict()),
                "rng_state": capture_rng(generator), "best_model_state": best_state,
                "epoch": epoch + 1, "best_epoch": best_epoch, "best_metric": best_metric,
                "stale_epochs": stale_epochs, "history": history, "config": config,
                "config_digest": config_digest(config), "split_digest": manifest["split_digest"],
                "bank_digest": None, "provenance": provenance, "model_config": model_config,
                "device": str(device)}
        atomic_torch_save(output / "last.pt", last)
        atomic_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            return {"status": "interrupted_at_epoch_boundary", "epoch": epoch + 1, "last": str(output / "last.pt")}
    if best_state is None:
        raise ValueError("No trained checkpoint is available")
    model.load_state_dict(best_state)
    manifest = load_manifest(manifest_path, verify_splits=("calibration",))
    calibration_records = [r for r in manifest["splits"]["calibration"] if r["label"] == 0]
    calibration = calibrate(model, calibration_records, config["preprocess"], config["batch_size"],
                            device, config["threshold_quantile"])
    checkpoint = {"schema_version": 2, "resumable": False, "model_kind": MODEL_KIND,
                  "model_state": best_state, "model_config": model_config, "category": "kolektor_surface",
                  "preprocess": config["preprocess"], "config": config, "split_digest": manifest["split_digest"],
                  "manifest_digest": manifest["digest"], "threshold": calibration["threshold"],
                  "threshold_source": "separate normal-only subset of calibration split",
                  "score_definition": SCORE_DEFINITION, "calibration": calibration,
                  "best_epoch": best_epoch, "best_metric": best_metric, "history": history,
                  "early_stopped": stopped_early, "provenance": provenance, "device": str(device),
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                  "training_kind": "random_initialization_real_defect_supervision",
                  "selection_rule": "maximize harmonic mean of validation image AUROC and model-space pixel AP",
                  "unused_positive_calibration_images": len(manifest["splits"]["calibration"]) - len(calibration_records)}
    atomic_torch_save(output / "checkpoint.pt", checkpoint)
    atomic_json(output / "summary.json", {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")})
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/ksdd2-manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/ksdd2_supervised.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--training-seed", type=int)
    args = parser.parse_args()
    train(args.manifest, args.config, args.output, args.resume, args.stop_after_epoch, args.device, args.training_seed)


if __name__ == "__main__":
    main()
