"""Isolated resumable trainer for the declared three-run screen."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inspection.checkpointing import (atomic_json, atomic_torch_save, capture_rng, collect_provenance,
                                     config_digest, cpu_copy, prepare_output, restore_rng, validate_resume)
from inspection.data import InspectionDataset, load_manifest
from inspection.model import JOINT_MODEL_KIND, SCORE_DEFINITION, create_model
from inspection.train import select_device, _loss_epoch, _synchronize, calibrate

from .bank import evaluate, load_bank, validate_split_sources
from .protocol import PROTOCOL, SELECTION_RULE, source_hashes
from .synthesis import synthesize


class SyntheticDataset(Dataset):
    """Both recipient and donor are from the same audited training-only pool."""
    def __init__(self, manifest, config):
        validate_split_sources(manifest)
        self.records = manifest["splits"]["train"]
        self.dataset = InspectionDataset(self.records, config["image_size"], include_masks=False)
        self.clean = [self.dataset[i]["image"] for i in range(len(self.dataset))]
        self.config, self.epoch = config, 0

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.config["training_seed"] + self.epoch * len(self) + index)
        donor_index = int(torch.randint(len(self) - 1, (), generator=generator))
        donor_index += donor_index >= index  # donor cannot be the recipient
        image, mask = synthesize(self.clean[index], self.clean[donor_index], generator,
                                 self.config["candidate"], self.config["normal_probability"])
        return {"image": image, "clean": self.clean[index], "mask": mask,
                "source_sha256": self.records[index]["image_sha256"],
                "donor_sha256": self.records[donor_index]["image_sha256"]}


def validate_config(config):
    if config["protocol"] != PROTOCOL or config["candidate"] not in ("control", "A", "B"):
        raise ValueError("Unsupported declared screen configuration")
    if config["model_kind"] != JOINT_MODEL_KIND or config["category"] != "metal_nut":
        raise ValueError("Only the declared metal_nut joint-model screen is implemented")
    if not 1 <= config["epochs"] <= 100 or not 1 <= config["selection_every"] <= config["epochs"]:
        raise ValueError("Invalid bounded epoch schedule")
    if (config["image_size"] < 16 or config["base_channels"] < 1 or config["batch_size"] < 1
            or config["cpu_threads"] < 1 or config["learning_rate"] <= 0
            or not 0 <= config["normal_probability"] <= 1 or config["threshold_quantile"] != .9):
        raise ValueError("Invalid architecture, optimizer or frozen calibration settings")


def validate_checkpoint(checkpoint, config, manifest, bank, declaration_digest, device, *, resumable):
    if checkpoint.get("declaration_digest") != declaration_digest:
        raise ValueError("Checkpoint declaration differs")
    if checkpoint["provenance"].get("v3_source_files") != source_hashes():
        raise ValueError("V3 source changed since declaration/checkpoint")
    # Shared validation checks source, dependencies, RNG environment and device.
    compatible = dict(checkpoint)
    if not resumable:
        if checkpoint.get("checkpoint_kind") != "inference" or checkpoint.get("resumable"):
            raise ValueError("Expected completed inference checkpoint")
        compatible["resumable"] = True
        if checkpoint.get("completed_epochs") != config["epochs"]:
            raise ValueError("Incomplete run cannot be reused as a finished candidate")
    validate_resume(compatible, config, manifest["split_digest"], JOINT_MODEL_KIND,
                    bank["bank_digest"], device)


def train(manifest_path, bank_path, config, output, declaration_digest, *, resume=None,
          stop_after_epoch=None, expected_sources=None):
    validate_config(config)
    if expected_sources is None or source_hashes() != expected_sources:
        raise ValueError("Training requires the unchanged source snapshot from a frozen declaration")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config["epochs"]:
        raise ValueError("Invalid stop_after_epoch")
    torch.set_num_threads(config["cpu_threads"])
    device = select_device(config["device"])
    manifest = load_manifest(manifest_path, verify_splits=("train", "validation", "calibration"))
    validate_split_sources(manifest)
    if manifest["config"]["seed"] != config["data_seed"]:
        raise ValueError("Original data split seed differs")
    if any(r["category"] != config["category"] for name in ("train", "validation", "calibration")
           for r in manifest["splits"][name]):
        raise ValueError("Manifest category differs")
    bank = load_bank(bank_path)
    if bank["split_digest"] != manifest["split_digest"] or bank["image_size"] != config["image_size"]:
        raise ValueError("Bank split or image resolution differs")
    output = Path(output)
    if (output / "checkpoint.pt").exists():
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, config, manifest, bank, declaration_digest, device, resumable=False)
        summary = {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")}
        summary_path = output / "summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
            raise ValueError("Completed summary differs from its checkpoint")
        if not summary_path.exists():  # recover a crash between the two atomic writes
            atomic_json(summary_path, summary)
        return checkpoint
    if resume is not None and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use this run's atomic last.pt")
    output = prepare_output(output, resume)
    seed = config["training_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    restored = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if restored:
        validate_checkpoint(restored, config, manifest, bank, declaration_digest, device, resumable=True)
    provenance = restored["provenance"] if restored else collect_provenance(config, manifest, bank["bank_digest"])
    provenance = {**provenance, "v3_source_files": expected_sources}
    training = SyntheticDataset(manifest, config)
    shuffle = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                        generator=shuffle, num_workers=0)
    model_config = {"base_channels": config["base_channels"]}
    model = create_model(JOINT_MODEL_KIND, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, best_state, best_metric, best_epoch, best_validation = [], None, None, None, None
    start, previous_elapsed = 0, 0.
    if restored:
        model.load_state_dict(restored["model_state"])
        optimizer.load_state_dict(restored["optimizer_state"])
        history, start, previous_elapsed = restored["history"], restored["epoch"], restored["elapsed_seconds"]
        best_state, best_metric = restored["best_model_state"], restored["best_metric"]
        best_epoch, best_validation = restored["best_epoch"], restored["best_validation"]
        restore_rng(restored["rng_state"], shuffle)
    common = {"schema_version": 2, "model_kind": JOINT_MODEL_KIND, "model_config": model_config,
              "image_size": config["image_size"], "config": config, "config_digest": config_digest(config),
              "preprocess": {"mode": "square", "height": config["image_size"], "width": config["image_size"]},
              "manifest_digest": manifest["digest"], "split_digest": manifest["split_digest"],
              "bank_digest": bank["bank_digest"], "declaration_digest": declaration_digest,
              "provenance": provenance, "device": str(device), "torch_version": str(torch.__version__),
              "score_definition": SCORE_DEFINITION, "model_selection": SELECTION_RULE,
              "training_kind": "random_initialization_normal_only_v3_synthetic_defects",
              "test_status": "development-inspected/exploratory; never used for selection"}
    atomic_json(output / "config.json", config)
    atomic_json(output / "provenance.json", provenance)
    started = time.perf_counter()
    print(json.dumps({"candidate": config["candidate"], "resumed_epoch": start, "device": str(device)}), flush=True)
    for epoch in range(start, config["epochs"]):
        training.epoch = epoch
        _synchronize(device)
        epoch_started = time.perf_counter()
        losses, seen, steps = _loss_epoch(model, loader, device, optimizer)
        eligible = (epoch + 1) % config["selection_every"] == 0 or epoch + 1 == config["epochs"]
        metrics = evaluate(model, bank_path, device, config["batch_size"]) if eligible else None
        improved = metrics is not None and (best_metric is None or metrics["macro_h"] > best_metric)
        if improved:
            best_state, best_metric = cpu_copy(model.state_dict()), metrics["macro_h"]
            best_epoch, best_validation = epoch + 1, metrics
        _synchronize(device)
        row = {"epoch": epoch + 1, "train": losses, "validation": metrics, "images_seen": seen,
               "steps": steps, "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        checkpoint = {**common, "checkpoint_kind": "last", "resumable": True, "epoch": epoch + 1,
                      "model_state": cpu_copy(model.state_dict()), "optimizer_state": cpu_copy(optimizer.state_dict()),
                      "rng_state": capture_rng(shuffle), "best_model_state": best_state,
                      "best_metric": best_metric, "best_epoch": best_epoch, "best_validation": best_validation,
                      "history": history, "threshold": None,
                      "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
        atomic_torch_save(output / "last.pt", checkpoint)
        if improved:
            atomic_torch_save(output / "best-uncalibrated.pt", checkpoint)
        atomic_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < config["epochs"]:
            return checkpoint
    if best_state is None:
        raise ValueError("No checkpoint selected by validation")
    model.load_state_dict(best_state)
    threshold, scores = calibrate(model, manifest["splits"]["calibration"], config["image_size"],
                                  config["batch_size"], device, config["threshold_quantile"])
    calibration = {"split": "calibration", "quantile": .9, "quantile_method": "linear", "count": len(scores),
                   "scores": scores, "threshold": threshold, "decision_rule": ">=",
                   "warning": "Normal calibration quantile is not a future false-alarm guarantee."}
    checkpoint = {**common, "checkpoint_kind": "inference", "resumable": False,
                  "completed_epochs": config["epochs"], "model_state": best_state,
                  "best_metric": best_metric, "best_epoch": best_epoch, "best_validation": best_validation,
                  "threshold": threshold, "threshold_source": "separate normal calibration split",
                  "calibration": calibration, "history": history,
                  "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
    atomic_torch_save(output / "checkpoint.pt", checkpoint)
    atomic_json(output / "summary.json", {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")})
    return checkpoint
