"""Train random-initialized networks without reading the real defect test split."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inspection.data import InspectionDataset, load_manifest
from inspection.losses import inspection_loss
from inspection.model import InspectionModel, anomaly_score
from inspection.synthesis import synthesize


class SyntheticDataset(Dataset):
    def __init__(self, records, image_size, seed, normal_probability, synthesis="legacy"):
        self.clean = InspectionDataset(records, image_size, include_masks=False)
        self.seed = seed
        self.epoch = 0
        self.normal_probability = normal_probability
        self.synthesis = synthesis

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean = self.clean[index]["image"]
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
        image, mask = synthesize(clean, generator, self.normal_probability, strategy=self.synthesis)
        return {"image": image, "clean": clean, "mask": mask}


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


def _loss_epoch(model, loader, device, optimizer=None, max_steps=None):
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "reconstruction": 0.0, "segmentation_bce": 0.0,
              "segmentation_dice": 0.0}
    seen = 0
    steps = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            image, clean, mask = [batch[key].to(device) for key in ("image", "clean", "mask")]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstruction, logits = model(image)
            loss, components = inspection_loss(reconstruction, logits, clean, mask)
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
    loader = DataLoader(InspectionDataset(records, image_size, include_masks=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    for batch in loader:
        _, logits = model(batch["image"].to(device))
        scores.extend(anomaly_score(logits).cpu().tolist())
    if not scores:
        raise ValueError("A separate nonempty normal calibration split is required")
    threshold = float(np.quantile(scores, quantile, method="linear"))
    return threshold, scores


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def train(manifest_path, config_path, output, epochs=None, max_steps=None, device_override=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if epochs is not None:
        config["epochs"] = epochs
    if device_override is not None:
        config["device"] = device_override
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config["image_size"] < 16 or config["base_channels"] < 1:
        raise ValueError("image_size must be at least 16 and base_channels must be positive")
    if not 0 <= config["normal_probability"] <= 1:
        raise ValueError("normal_probability must be between zero and one")
    if config.get("synthesis", "legacy") not in ("legacy", "foreground"):
        raise ValueError("synthesis must be legacy or foreground")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not 0 < config["threshold_quantile"] < 1:
        raise ValueError("threshold_quantile must be between zero and one")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Repeated CPU runs are reproducible; MPS/CUDA kernels can still vary.
    device = select_device(config.get("device", "auto"))
    manifest = load_manifest(
        manifest_path,
        verify_splits=("train", "validation", "calibration"),
    )
    splits = manifest["splits"]
    for name in ("train", "validation", "calibration"):
        records = splits[name]
        if not records or any(record["label"] != 0 for record in records):
            raise ValueError(f"{name} must contain only normal images and must be nonempty")
        if any(record["category"] != config["category"] for record in records):
            raise ValueError(f"{name} category does not match the training configuration")
    normal_paths = [{record["image"] for record in splits[name]}
                    for name in ("train", "validation", "calibration")]
    if any(normal_paths[i] & normal_paths[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Normal train, validation and calibration images must be disjoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists():
        raise FileExistsError("Output already has a checkpoint; choose a new run directory")
    _write_json(output / "config.json", config)
    size, batch_size = config["image_size"], config["batch_size"]
    synthesis = config.get("synthesis", "legacy")
    training = SyntheticDataset(splits["train"], size, seed, config["normal_probability"], synthesis)
    validation = SyntheticDataset(splits["validation"], size, seed + 1_000_000,
                                  config["normal_probability"], synthesis)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              generator=shuffle_generator, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    model_config = {"base_channels": config["base_channels"]}
    model = InspectionModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    started = time.perf_counter()
    print(json.dumps({"device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
                      "train_images": len(training), "validation_images": len(validation),
                      "calibration_images": len(splits["calibration"]), "max_steps": max_steps}), flush=True)
    for epoch in range(config["epochs"]):
        training.epoch = epoch
        _synchronize(device)
        epoch_started = time.perf_counter()
        train_metrics, seen, steps = _loss_epoch(model, train_loader, device, optimizer, max_steps)
        validation_metrics, _, _ = _loss_epoch(model, validation_loader, device)
        _synchronize(device)
        row = {"epoch": epoch + 1, "train": train_metrics, "validation": validation_metrics,
               "train_images_seen": seen, "steps": steps,
               "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        _write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    threshold, normal_scores = calibrate(model, splits["calibration"], size, batch_size, device,
                                         config["threshold_quantile"])
    _synchronize(device)
    calibration = {
        "split": "calibration", "count": len(normal_scores),
        "quantile": config["threshold_quantile"], "threshold": threshold,
        "scores": normal_scores,
        "quantile_method": "linear", "decision_rule": ">=",
        "calibration_exceedances": int(sum(score >= threshold for score in normal_scores)),
        "score_std": float(np.std(normal_scores)),
        "warning": "Empirical normal quantile; small calibration sets do not guarantee future false alarm rates.",
    }
    checkpoint = {
        "model_state": best_state, "model_config": model_config, "image_size": size,
        "threshold": threshold, "manifest_digest": manifest["digest"],
        "split_digest": manifest["split_digest"], "config": config,
        "history": history, "best_epoch": best_epoch, "calibration": calibration,
        "device": str(device), "torch_version": str(torch.__version__),
        "threshold_source": "separate normal calibration split",
        "elapsed_seconds": time.perf_counter() - started,
        "training_kind": "random_initialization_procedural_synthetic_defects",
        "smoke_run": max_steps is not None,
    }
    temporary = output / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output / "checkpoint.pt")
    summary = {key: value for key, value in checkpoint.items() if key not in ("model_state", "history")}
    _write_json(output / "summary.json", summary)
    print(json.dumps({"checkpoint": str(output / "checkpoint.pt"), "best_epoch": best_epoch,
                      "threshold": threshold, "elapsed_seconds": checkpoint["elapsed_seconds"]}), flush=True)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/metal_nut.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, help="Limit training batches per epoch for a smoke run")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    train(args.manifest, args.config, args.output, args.epochs, args.max_steps, args.device)


if __name__ == "__main__":
    main()
