"""Train random-initialized models with atomic epoch checkpoints and exact CPU resume."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from inspection.checkpointing import (
    SCHEMA_VERSION, atomic_json, atomic_torch_save, capture_rng, collect_provenance,
    config_digest, cpu_copy, prepare_output, restore_rng, validate_resume,
)
from inspection.data import InspectionDataset, load_manifest
from inspection.losses import inspection_loss
from inspection.model import (
    JOINT_MODEL_KIND, SEGMENTATION_MODEL_KIND, SCORE_DEFINITION, anomaly_score, create_model,
)
from inspection.synthesis import synthesize

BASELINE_KIND = "normal_only_denoising_reconstruction"
BASELINE_SCORE_DEFINITION = "mean top 1% of channel-mean absolute reconstruction error; not a probability"


class CachedNormalDataset(Dataset):
    """Decode once; cached images are immutable inputs to deterministic augmentation."""
    def __init__(self, records, image_size, cache_images=True):
        self.dataset = InspectionDataset(records, image_size, include_masks=False)
        self.cache = [self.dataset[index] for index in range(len(self.dataset))] if cache_images else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.cache[index] if self.cache is not None else self.dataset[index]


class SyntheticDataset(Dataset):
    def __init__(self, records, image_size, seed, normal_probability, synthesis="legacy",
                 restrict_foreground=True, scratch_enabled=True, cache_images=True):
        self.clean = CachedNormalDataset(records, image_size, cache_images)
        self.seed, self.epoch = seed, 0
        self.normal_probability, self.synthesis = normal_probability, synthesis
        self.restrict_foreground, self.scratch_enabled = restrict_foreground, scratch_enabled

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean = self.clean[index]["image"]
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
        image, mask = synthesize(clean, generator, self.normal_probability, strategy=self.synthesis,
                                 restrict_foreground=self.restrict_foreground,
                                 scratch_enabled=self.scratch_enabled)
        return {"image": image, "clean": clean, "mask": mask}


class DenoisingDataset(Dataset):
    def __init__(self, records, image_size, noise_std=0, cache_images=True):
        self.clean = CachedNormalDataset(records, image_size, cache_images)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean = self.clean[index]["image"]
        image = (clean + torch.randn_like(clean) * self.noise_std).clamp(0, 1) if self.noise_std else clean
        return {"image": image, "clean": clean, "mask": torch.zeros_like(clean[:1])}


def select_device(name):
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable in this PyTorch environment")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _score(second_output, model_kind):
    if model_kind == BASELINE_KIND:
        from inspection.baseline import reconstruction_score
        return reconstruction_score(second_output)
    return anomaly_score(second_output)


def _loss_epoch(model, loader, device, optimizer=None, max_steps=None):
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "reconstruction": 0.0, "segmentation_bce": 0.0, "segmentation_dice": 0.0}
    seen, steps = 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            image, clean, mask = [batch[key].to(device) for key in ("image", "clean", "mask")]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, logits = model(image)
            if model.model_kind == BASELINE_KIND:
                loss = F.l1_loss(reconstruction, clean)
                components = {"reconstruction": loss.detach()}
            else:
                loss, components = inspection_loss(reconstruction, logits, clean, mask)
                if model.model_kind == SEGMENTATION_MODEL_KIND:
                    loss = loss - components["reconstruction"]
                    components["reconstruction"] = loss.new_zeros(())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing to save an invalid model")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            size = image.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * size
            for key, value in components.items():
                totals[key] += float(value.cpu()) * size
            seen += size
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
    if not seen:
        raise ValueError("Empty training or validation dataset")
    return {key: value / seen for key, value in totals.items()}, seen, steps


@torch.no_grad()
def calibrate(model, records, image_size, batch_size, device, quantile):
    model.eval()
    loader = DataLoader(CachedNormalDataset(records, image_size), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    scores = []
    for batch in loader:
        _, second = model(batch["image"].to(device))
        scores.extend(_score(second, model.model_kind).cpu().tolist())
    if not scores or not np.isfinite(scores).all():
        raise ValueError("A nonempty normal calibration split with finite scores is required")
    return float(np.quantile(scores, quantile, method="linear")), scores


def _normalized_config(config, epochs=None, device_override=None, model_kind_override=None):
    config = dict(config)
    if epochs is not None:
        config["epochs"] = epochs
    if device_override is not None:
        config["device"] = device_override
    if model_kind_override is not None:
        config["model_kind"] = model_kind_override
    config.setdefault("model_kind", JOINT_MODEL_KIND)
    config["training_seed"] = config.pop("seed", config.get("training_seed", 42))
    config.setdefault("cache_images", True)
    config.setdefault("cpu_threads", 1)
    config.setdefault("selection_every", 1)
    config.setdefault("require_bank", False)
    config.setdefault("device", "auto")
    if config["model_kind"] != BASELINE_KIND:
        config.setdefault("synthesis", "legacy")
        config.setdefault("restrict_foreground", True)
        config.setdefault("scratch_enabled", True)
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["cpu_threads"] < 1:
        raise ValueError("epochs, batch_size and cpu_threads must be positive")
    if config["image_size"] < 16 or config["base_channels"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("image_size>=16 and positive base_channels and learning_rate required")
    if not 0 < config["threshold_quantile"] < 1 or config["selection_every"] < 1:
        raise ValueError("Invalid threshold_quantile or selection_every")
    if config["model_kind"] == BASELINE_KIND:
        if not 0 <= config.get("noise_std", 0) <= 0.2:
            raise ValueError("noise_std must be between zero and 0.2")
    elif config["model_kind"] in (JOINT_MODEL_KIND, SEGMENTATION_MODEL_KIND):
        if not 0 <= config["normal_probability"] <= 1 or config["synthesis"] not in ("legacy", "foreground"):
            raise ValueError("Invalid normal_probability or synthesis")
    else:
        raise ValueError("Unsupported model kind")
    return config


def train(manifest_path, config_path, output, epochs=None, max_steps=None, device_override=None,
          resume=None, bank_path=None, stop_after_epoch=None, model_kind_override=None):
    config = _normalized_config(json.loads(Path(config_path).read_text(encoding="utf-8")),
                                epochs, device_override, model_kind_override)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("stop_after_epoch must be within the declared epoch budget")
    if config["require_bank"] and bank_path is None:
        raise ValueError("This declared experiment requires --bank with the immutable validation challenge bank")
    torch.set_num_threads(config["cpu_threads"])
    seed = config["training_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(config["device"])
    manifest = load_manifest(manifest_path, verify_splits=("train", "validation", "calibration"))
    if "data_seed" in config and config["data_seed"] != manifest["config"]["seed"]:
        raise ValueError("The declared data_seed differs from the audited split")
    splits = manifest["splits"]
    for name in ("train", "validation", "calibration"):
        if not splits[name] or any(r["label"] != 0 or r["category"] != config["category"] for r in splits[name]):
            raise ValueError(f"{name} must be nonempty normal-only data for the configured category")
    paths = [{r["image"] for r in splits[name]} for name in ("train", "validation", "calibration")]
    if any(paths[i] & paths[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Normal training, validation and calibration images must be disjoint")
    bank_digest = None
    if bank_path is not None:
        from inspection.validation_bank import load_bank
        bank = load_bank(bank_path)
        bank_digest = bank["bank_digest"]
        if bank["split_digest"] != manifest["split_digest"]:
            raise ValueError("Challenge bank and training split differ")
    output = prepare_output(output, resume)
    restored = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if restored is not None:
        validate_resume(restored, config, manifest["split_digest"], config["model_kind"], bank_digest, device)
        if restored.get("max_steps") != max_steps:
            raise ValueError("Resume max_steps differs")
    atomic_json(output / "config.json", config)
    provenance = restored["provenance"] if restored else collect_provenance(config, manifest, bank_digest)
    atomic_json(output / "provenance.json", provenance)
    size, batch_size = config["image_size"], config["batch_size"]
    if config["model_kind"] == BASELINE_KIND:
        training = DenoisingDataset(splits["train"], size, config.get("noise_std", 0), config["cache_images"])
        validation = DenoisingDataset(splits["validation"], size, 0, config["cache_images"])
    else:
        options = {key: config[key] for key in ("restrict_foreground", "scratch_enabled", "cache_images")}
        training = SyntheticDataset(splits["train"], size, seed, config["normal_probability"], config["synthesis"], **options)
        validation = SyntheticDataset(splits["validation"], size, seed + 1_000_000,
                                      config["normal_probability"], config["synthesis"], **options)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=shuffle_generator, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    model_config = {"base_channels": config["base_channels"]}
    model = create_model(config["model_kind"], model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, best_state, best_epoch, best_metric = [], None, None, None
    start_epoch, previous_elapsed = 0, 0.0
    if restored:
        model.load_state_dict(restored["model_state"])
        optimizer.load_state_dict(restored["optimizer_state"])
        history = restored["history"]
        best_state, best_epoch, best_metric = restored["best_model_state"], restored["best_epoch"], restored["best_metric"]
        start_epoch, previous_elapsed = restored["epoch"], restored["elapsed_seconds"]
        restore_rng(restored["rng_state"], shuffle_generator)
    started = time.perf_counter()
    selection_rule = "maximum common synthetic challenge harmonic mean(image AUROC, pixel AP)" if bank_path else "minimum fixed validation loss"
    common = {
        "schema_version": SCHEMA_VERSION, "model_kind": config["model_kind"], "model_config": model_config,
        "image_size": size, "preprocess": {"mode": "square", "height": size, "width": size},
        "config": config, "config_digest": config_digest(config), "manifest_digest": manifest["digest"],
        "split_digest": manifest["split_digest"], "bank_digest": bank_digest, "provenance": provenance,
        "device": str(device), "torch_version": str(torch.__version__), "max_steps": max_steps,
        "score_definition": BASELINE_SCORE_DEFINITION if config["model_kind"] == BASELINE_KIND else SCORE_DEFINITION,
        "model_selection": selection_rule, "smoke_run": max_steps is not None,
        "training_kind": "random_initialization_normal_denoising" if config["model_kind"] == BASELINE_KIND else "random_initialization_procedural_synthetic_defects",
    }
    print(json.dumps({"device": str(device), "model_kind": config["model_kind"], "resumed_epoch": start_epoch,
                      "parameters": sum(p.numel() for p in model.parameters()), "max_steps": max_steps,
                      "train_images": len(training), "validation_images": len(validation),
                      "selection_rule": selection_rule}), flush=True)
    last = restored
    for epoch in range(start_epoch, config["epochs"]):
        if isinstance(training, SyntheticDataset):
            training.epoch = epoch
        _synchronize(device)
        epoch_started = time.perf_counter()
        train_metrics, seen, steps = _loss_epoch(model, train_loader, device, optimizer, max_steps)
        validation_metrics, _, _ = _loss_epoch(model, validation_loader, device)
        eligible = (epoch + 1) % config["selection_every"] == 0 or epoch + 1 == config["epochs"]
        challenge = None
        improved = False
        if eligible:
            if bank_path is not None:
                from inspection.validation_bank import evaluate_bank
                challenge = evaluate_bank(model, config["model_kind"], bank_path, device, size, batch_size)
                metric = challenge["selection_score"]
                if metric is None or not np.isfinite(metric):
                    raise ValueError("Challenge bank produced an invalid selection score")
                improved = best_metric is None or metric > best_metric
            else:
                metric = validation_metrics["loss"]
                improved = best_metric is None or metric < best_metric
            if improved:
                best_metric, best_epoch = float(metric), epoch + 1
                best_state = cpu_copy(model.state_dict())
        _synchronize(device)
        row = {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics,
               "challenge": challenge, "train_images_seen": seen, "steps": steps,
               "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        last = {**common, "checkpoint_kind": "last", "resumable": True,
                "epoch": epoch + 1, "model_state": cpu_copy(model.state_dict()),
                "optimizer_state": cpu_copy(optimizer.state_dict()), "rng_state": capture_rng(shuffle_generator),
                "best_model_state": best_state, "best_epoch": best_epoch, "best_metric": best_metric,
                "history": history, "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                "threshold": None}
        atomic_torch_save(output / "last.pt", last)
        if improved:
            atomic_torch_save(output / "best.pt", {**last, "checkpoint_kind": "best"})
        atomic_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < config["epochs"]:
            print(json.dumps({"paused_after_epoch": epoch + 1, "resume": str(output / "last.pt")}), flush=True)
            return last
    if best_state is None:
        raise ValueError("No checkpoint was eligible for selection")
    model.load_state_dict(best_state)
    threshold, scores = calibrate(model, splits["calibration"], size, batch_size, device, config["threshold_quantile"])
    _synchronize(device)
    calibration = {
        "split": "calibration", "count": len(scores), "quantile": config["threshold_quantile"],
        "threshold": threshold, "scores": scores, "quantile_method": "linear", "decision_rule": ">=",
        "calibration_exceedances": int(sum(score >= threshold for score in scores)), "score_std": float(np.std(scores)),
        "warning": "Empirical normal quantile; finite calibration sets do not guarantee future false-alarm rates.",
    }
    checkpoint = {**common, "checkpoint_kind": "inference", "resumable": False,
                  "model_state": best_state, "history": history, "best_epoch": best_epoch,
                  "best_metric": best_metric, "threshold": threshold, "calibration": calibration,
                  "threshold_source": "separate normal calibration split",
                  "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
    atomic_torch_save(output / "checkpoint.pt", checkpoint)
    atomic_torch_save(output / "best.pt", checkpoint)
    atomic_json(output / "summary.json", {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")})
    print(json.dumps({"checkpoint": str(output / "checkpoint.pt"), "best_epoch": best_epoch,
                      "threshold": threshold, "elapsed_seconds": checkpoint["elapsed_seconds"]}), flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/metal_nut.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--resume", help="Resume this run's atomic last.pt with unchanged configuration")
    parser.add_argument("--bank", help="Immutable common synthetic validation challenge bank.json")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop at an epoch boundary without final calibration")
    args = parser.parse_args()
    train(args.manifest, args.config, args.output, args.epochs, args.max_steps, args.device,
          args.resume, args.bank, args.stop_after_epoch)


if __name__ == "__main__":
    main()
