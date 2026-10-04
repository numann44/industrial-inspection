"""Two matched supervised runs with resumable selection, then gated calibration."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from inspection.checkpointing import (atomic_json, atomic_torch_save, capture_rng, collect_provenance,
                                     config_digest, cpu_copy, prepare_output, restore_rng, validate_resume)
from inspection.ksdd2 import load_manifest
from inspection.model import SegmentationOnlyModel
from inspection.supervised import (MODEL_KIND, SCORE_DEFINITION, BalancedBatchSampler, calibrate,
                                   segmentation_loss)
from inspection.train import select_device, _synchronize

from .data import TrainingDataset, load_bank, require_isolated_splits, verify_bank_membership, verify_records
from .metrics import evaluate
from .protocol import PROTOCOL, SELECTION_RULE, check_identity, source_hashes, sha256


def validate_config(cfg):
    if cfg["protocol"] != PROTOCOL or cfg["candidate"] not in ("control", "acquisition_aug"):
        raise ValueError("Unknown robustness screen configuration")
    if (not 1 <= cfg["epochs"] <= 100 or not 1 <= cfg["selection_every"] <= cfg["epochs"]
            or cfg["patience_epochs"] < cfg["selection_every"]
            or cfg["patience_epochs"] % cfg["selection_every"] != 0):
        raise ValueError("Patience must be whole validation intervals measured in training epochs")
    if (cfg["batch_size"] < 2 or cfg["batch_size"] % 2 or cfg["base_channels"] < 1
            or cfg["learning_rate"] <= 0 or cfg["positive_pixel_weight"] != 3.
            or cfg["threshold_quantile"] != .9 or cfg["cpu_threads"] != 1):
        raise ValueError("Invalid matched optimization/calibration configuration")
    if cfg["preprocess"]["mode"] != "letterbox" or min(cfg["preprocess"]["height"], cfg["preprocess"]["width"]) < 16:
        raise ValueError("Expected aspect-preserving letterbox preprocessing")


def validate_checkpoint(checkpoint, cfg, manifest, bank, declaration, device, *, selected=False):
    check_identity(declaration)
    if checkpoint.get("declaration_digest") != declaration["declaration_digest"]:
        raise ValueError("Checkpoint declaration differs")
    if checkpoint["provenance"].get("robustness_source_files") != source_hashes():
        raise ValueError("Robustness checkpoint source differs")
    compatible = dict(checkpoint)
    if selected:
        if checkpoint.get("checkpoint_kind") != "validation_selected" or checkpoint.get("resumable"):
            raise ValueError("Expected an immutable validation-selected checkpoint")
        ended = checkpoint["completed_epochs"] == cfg["epochs"] or (
            checkpoint["early_stopped"] and checkpoint["stale_epochs"] >= cfg["patience_epochs"])
        if not ended:
            raise ValueError("Incomplete run cannot be reused as a selected candidate")
        compatible["resumable"] = True
    validate_resume(compatible, cfg, manifest["split_digest"], MODEL_KIND, bank["bank_digest"], device)


def train(manifest_path, bank_path, cfg, output, declaration, *, resume=None, stop_after_epoch=None):
    """Never reads calibration or test pixels, including on a completed-run reuse."""
    validate_config(cfg)
    check_identity(declaration)
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= cfg["epochs"]:
        raise ValueError("Invalid interruption epoch")
    torch.set_num_threads(cfg["cpu_threads"])
    device = select_device(cfg["device"])
    manifest = load_manifest(manifest_path, verify_splits=("train", "validation"))
    require_isolated_splits(manifest)
    bank = load_bank(bank_path)
    verify_bank_membership(bank, manifest)
    if bank["preprocess"] != cfg["preprocess"]:
        raise ValueError("Validation preprocessing differs")
    output = Path(output)
    if (output / "selected.pt").exists():
        selected = torch.load(output / "selected.pt", map_location="cpu", weights_only=True)
        validate_checkpoint(selected, cfg, manifest, bank, declaration, device, selected=True)
        summary = {k: v for k, v in selected.items() if k not in ("model_state", "history")}
        summary_path = output / "summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
            raise ValueError("Completed summary differs from selected checkpoint")
        if not summary_path.exists():
            atomic_json(summary_path, summary)
        return selected
    if resume is not None and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Can resume only this run's last.pt")
    output = prepare_output(output, resume)
    seed = cfg["training_seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    previous = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if previous:
        validate_checkpoint(previous, cfg, manifest, bank, declaration, device)
    provenance = previous["provenance"] if previous else collect_provenance(cfg, manifest, bank["bank_digest"])
    provenance = {**provenance, "robustness_source_files": declaration["source_files"]}
    training = TrainingDataset(manifest["splits"]["train"], cfg)
    sampler_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_sampler=BalancedBatchSampler(training.records, cfg["batch_size"], sampler_generator),
                        num_workers=0)
    model_config = {"base_channels": cfg["base_channels"]}
    model = SegmentationOnlyModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    history, best_state, best_metrics, best_epoch, best_metric = [], None, None, None, None
    first_epoch, stale_epochs, previous_elapsed = 0, 0, 0.
    if previous:
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        history, first_epoch = previous["history"], previous["epoch"]
        best_state, best_metrics = previous["best_model_state"], previous["best_validation"]
        best_epoch, best_metric, stale_epochs = previous["best_epoch"], previous["best_metric"], previous["stale_epochs"]
        previous_elapsed = previous["elapsed_seconds"]
        restore_rng(previous["rng_state"], sampler_generator)
        training.generator.set_state(previous["rng_state"]["augmentation"].cpu())
    common = {"schema_version": 2, "model_kind": MODEL_KIND, "model_config": model_config,
              "category": "kolektor_surface", "preprocess": cfg["preprocess"], "config": cfg,
              "config_digest": config_digest(cfg), "manifest_digest": manifest["digest"],
              "split_digest": manifest["split_digest"], "bank_digest": bank["bank_digest"],
              "declaration_digest": declaration["declaration_digest"], "provenance": provenance,
              "device": str(device), "score_definition": SCORE_DEFINITION, "model_selection": SELECTION_RULE,
              "training_kind": "random_initialization_real_defect_supervision_acquisition_screen",
              "test_status": "development-inspected/exploratory for all reused KSDD2 tests"}
    atomic_json(output / "config.json", cfg)
    atomic_json(output / "provenance.json", provenance)
    started = time.perf_counter()
    print(json.dumps({"candidate": cfg["candidate"], "resumed_epoch": first_epoch, "device": str(device)}), flush=True)
    for epoch in range(first_epoch, cfg["epochs"]):
        if stale_epochs >= cfg["patience_epochs"]:
            break
        training.set_epoch(epoch)
        model.train()
        total_loss, seen = 0., 0
        _synchronize(device)
        epoch_start = time.perf_counter()
        for batch in loader:
            image, mask, valid = [batch[key].to(device) for key in ("image", "mask", "valid_mask")]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(image)
            loss, _ = segmentation_loss(logits, mask, valid, cfg["positive_pixel_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(image)
            seen += len(image)
        eligible = (epoch + 1) % cfg["selection_every"] == 0 or epoch + 1 == cfg["epochs"]
        metrics = evaluate(model, bank_path, device, cfg["batch_size"], output / "scratch") if eligible else None
        improved = metrics is not None and (best_metric is None or metrics["pooled"]["harmonic_mean"] > best_metric)
        if improved:
            best_state, best_epoch, best_metrics = cpu_copy(model.state_dict()), epoch + 1, metrics
            best_metric, stale_epochs = metrics["pooled"]["harmonic_mean"], 0
        elif eligible:
            # Three failed 5-epoch validation checks are 15 training epochs,
            # not 15 checks (75 epochs). Do not increment between checks.
            previous_check = max((row["epoch"] for row in history if row["validation"] is not None), default=0)
            stale_epochs += epoch + 1 - previous_check
        _synchronize(device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / seen, "images_seen": seen,
               "validation": metrics, "stale_epochs": stale_epochs,
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        rng = capture_rng(sampler_generator)
        rng["augmentation"] = training.generator.get_state().clone()
        last = {**common, "checkpoint_kind": "last", "resumable": True, "epoch": epoch + 1,
                "model_state": cpu_copy(model.state_dict()), "optimizer_state": cpu_copy(optimizer.state_dict()),
                "rng_state": rng, "best_model_state": best_state, "best_epoch": best_epoch,
                "best_metric": best_metric, "best_validation": best_metrics, "stale_epochs": stale_epochs,
                "history": history, "threshold": None, "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
        atomic_torch_save(output / "last.pt", last)
        if improved:
            atomic_torch_save(output / "best-uncalibrated.pt", last)
        atomic_json(output / "history.json", history)
        # Group predictions remain in the checkpoint/history, not repeated in
        # the console for every validation (keeps unattended logs readable).
        logged = {**row, "validation": None if metrics is None else {k: v for k, v in metrics.items()
                  if k not in ("group_scores", "group_labels", "group_ids")}}
        print(json.dumps(logged), flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < cfg["epochs"]:
            return last
    if best_state is None:
        raise ValueError("No checkpoint was eligible for validation selection")
    selected = {**common, "checkpoint_kind": "validation_selected", "resumable": False,
                "completed_epochs": len(history), "early_stopped": len(history) < cfg["epochs"],
                "stale_epochs": stale_epochs, "model_state": best_state, "best_epoch": best_epoch,
                "best_metric": best_metric, "best_validation": best_metrics, "history": history,
                "threshold": None, "calibration_status": "not read; requires recorded two-run gate pass",
                "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
    atomic_torch_save(output / "selected.pt", selected)
    atomic_json(output / "summary.json", {k: v for k, v in selected.items() if k not in ("model_state", "history")})
    return selected


def calibrate_after_gate(manifest_path, bank_path, cfg, output, declaration, gate_result):
    """Called only after the runner writes the frozen global gate; no feedback."""
    from .protocol import gate as compute_gate
    if gate_result != compute_gate(gate_result["control"], gate_result["candidate"]):
        raise ValueError("Gate record does not match the declared validation-only rule")
    if gate_result["state"] != "awaiting_exploratory_evaluation":
        raise ValueError("Failed gate forbids calibration and further work in this screen")
    output = Path(output)
    gate_path = output.parents[1] / "screen-result.json"
    if not gate_path.is_file() or json.loads(gate_path.read_text()) != gate_result:
        raise ValueError("Global gate must be recorded before any calibration read")
    check_identity(declaration)
    torch.set_num_threads(cfg["cpu_threads"])
    device = select_device(cfg["device"])
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path, verify_sources=False)
    selected_path = output / "selected.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    validate_checkpoint(selected, cfg, manifest, bank, declaration, device, selected=True)
    result_row = gate_result["control"] if cfg["candidate"] == "control" else gate_result["candidate"]
    selected_hash = sha256(selected_path)
    if result_row["checkpoint_sha256"] != selected_hash:
        raise ValueError("Selected weights changed after the global gate")
    identity = {"selected_checkpoint_sha256": selected_hash, "gate_digest": config_digest(gate_result)}
    checkpoint_path = output / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("calibration_identity") != identity or checkpoint.get("config_digest") != config_digest(cfg)
                or checkpoint.get("split_digest") != manifest["split_digest"] or checkpoint.get("bank_digest") != bank["bank_digest"]
                or checkpoint.get("checkpoint_kind") != "inference" or checkpoint.get("threshold") is None):
            raise ValueError("Existing calibrated checkpoint has incompatible identity")
        if (checkpoint["model_state"].keys() != selected["model_state"].keys()
                or any(not torch.equal(checkpoint["model_state"][key], value) for key, value in selected["model_state"].items())
                or not np.isfinite(checkpoint["threshold"])
                or checkpoint["threshold"] != checkpoint["calibration"]["threshold"]
                or checkpoint["calibration"]["quantile"] != .9
                or checkpoint["threshold"] != float(np.quantile(checkpoint["calibration"]["scores"], .9, method="linear"))):
            raise ValueError("Calibrated weights or threshold differ from frozen evidence")
        summary = {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")}
        summary_path = output / "calibration-summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
            raise ValueError("Calibrated summary differs from its checkpoint")
        if not summary_path.exists():
            atomic_json(summary_path, summary)
        return checkpoint
    normals = [row for row in manifest["splits"]["calibration"] if row["label"] == 0]
    verify_records(normals, include_masks=False)  # positive calibration images remain unopened
    model = SegmentationOnlyModel(**selected["model_config"]).to(device)
    model.load_state_dict(selected["model_state"])
    calibration = calibrate(model, normals, cfg["preprocess"], cfg["batch_size"], device, .9)
    checkpoint = {**selected, "checkpoint_kind": "inference", "threshold": calibration["threshold"],
                  "calibration": calibration, "calibration_identity": identity,
                  "calibration_status": "fit once after immutable two-run validation gate passed",
                  "threshold_source": "separate original clean normal calibration split"}
    atomic_torch_save(checkpoint_path, checkpoint)
    atomic_json(output / "calibration-summary.json", {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")})
    return checkpoint
