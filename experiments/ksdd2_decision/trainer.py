"""Two matched from-scratch runs: score-aware selection then gated q90 calibration."""
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
from torch.nn import functional as F
from inspection.supervised import BalancedBatchSampler, segmentation_loss
from inspection.train import select_device, _synchronize

from experiments.ksdd2_robustness.data import (TrainingDataset, load_bank, require_isolated_splits,
                                             verify_bank_membership, verify_records)
from .metrics import evaluate, calibrate
from .model import make_model, identity, parameter_counts, check_score_identity
from .protocol import PROTOCOL, SELECTION_RULE, check_identity, source_hashes, sha256


def validate_config(cfg):
    if cfg["protocol"] != PROTOCOL or cfg["candidate"] not in ("control", "decision_head"):
        raise ValueError("Unknown decision screen configuration")
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
    from .protocol import config
    reference = config(cfg["candidate"], cfg["device"])
    if any(cfg[key] != reference[key] for key in ("training_seed", "head_seed", "augmentation", "max_fpr",
                                                 "classification_loss_weight", "classification_gradients_to_segmentation")):
        raise ValueError("Decision seed, augmentation, scoring or gradient policy differs")


def validate_binding(declaration, cfg, manifest_path, bank_path, output):
    """Bind operational calls to the declared run; tiny fixtures use config only."""
    if "runs" in declaration:
        matches = [job for job in declaration["runs"] if job["config"] == cfg
                   and Path(job["output"]).resolve() == Path(output).resolve()]
        if len(matches) != 1:
            raise ValueError("Call is not one of the two declared run configurations/paths")
    elif declaration.get("config") != cfg:
        raise ValueError("Fixture configuration differs from declaration")
    if "inputs" in declaration:
        inputs = declaration["inputs"]
        if (Path(manifest_path).resolve() != Path(inputs["manifest"]).resolve()
                or Path(bank_path).resolve() != Path(inputs["bank"]).resolve()
                or sha256(manifest_path) != inputs["manifest_file_sha256"]
                or sha256(bank_path) != inputs["bank_file_sha256"]):
            raise ValueError("Manifest or validation bank differs from declaration")


def validate_checkpoint(checkpoint, cfg, manifest, bank, declaration, device, *, selected=False):
    check_identity(declaration)
    if checkpoint.get("declaration_digest") != declaration["declaration_digest"]:
        raise ValueError("Checkpoint declaration differs")
    if checkpoint["provenance"].get("decision_source_files") != source_hashes():
        raise ValueError("Decision checkpoint source differs")
    check_score_identity(checkpoint, cfg)
    compatible = dict(checkpoint)
    if selected:
        if checkpoint.get("checkpoint_kind") != "validation_selected" or checkpoint.get("resumable"):
            raise ValueError("Expected an immutable validation-selected checkpoint")
        ended = checkpoint["completed_epochs"] == cfg["epochs"] or (
            checkpoint["early_stopped"] and checkpoint["stale_epochs"] >= cfg["patience_epochs"])
        if not ended:
            raise ValueError("Incomplete run cannot be reused as a selected candidate")
        compatible["resumable"] = True
    validate_resume(compatible, cfg, manifest["split_digest"], identity(cfg["candidate"])[0], bank["bank_digest"], device)


def train(manifest_path, bank_path, cfg, output, declaration, *, resume=None, stop_after_epoch=None):
    """Never reads calibration or test pixels, including on a completed-run reuse."""
    validate_config(cfg)
    check_identity(declaration)
    validate_binding(declaration, cfg, manifest_path, bank_path, output)
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
    provenance = {**provenance, "decision_source_files": declaration["source_files"]}
    training = TrainingDataset(manifest["splits"]["train"], {**cfg, "candidate": "acquisition_aug"})
    sampler_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_sampler=BalancedBatchSampler(training.records, cfg["batch_size"], sampler_generator),
                        num_workers=0)
    model_config = {"base_channels": cfg["base_channels"]}
    model = make_model(cfg).to(device)
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
    common = {"schema_version": 2, "model_kind": model.model_kind, "model_config": model_config,
              "decision_model_config": {"candidate": cfg["candidate"], "base_channels": cfg["base_channels"], "head_seed": cfg["head_seed"]},
              "parameter_counts": parameter_counts(model),
              "category": "kolektor_surface", "preprocess": cfg["preprocess"], "config": cfg,
              "config_digest": config_digest(cfg), "manifest_digest": manifest["digest"],
              "split_digest": manifest["split_digest"], "bank_digest": bank["bank_digest"],
              "declaration_digest": declaration["declaration_digest"], "provenance": provenance,
              "device": str(device), "score_definition": model.score_definition, "model_selection": SELECTION_RULE,
              "training_kind": "random_initialization_real_defect_supervision_detached_decision_screen",
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
        total_loss, total_seg_loss, total_cls_loss, seen = 0., 0., 0., 0
        _synchronize(device)
        epoch_start = time.perf_counter()
        for batch in loader:
            image, mask, valid = [batch[key].to(device) for key in ("image", "mask", "valid_mask")]
            optimizer.zero_grad(set_to_none=True)
            logits, image_scores = model(image, valid)
            seg_loss, _ = segmentation_loss(logits, mask, valid, cfg["positive_pixel_weight"])
            cls_loss = (F.binary_cross_entropy_with_logits(image_scores, batch["label"].to(device).float())
                        if cfg["candidate"] == "decision_head" else image_scores.new_zeros(()))
            loss = seg_loss + cfg["classification_loss_weight"] * cls_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised loss")
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(image)
            total_seg_loss += float(seg_loss.detach().cpu()) * len(image)
            total_cls_loss += float(cls_loss.detach().cpu()) * len(image)
            seen += len(image)
        eligible = (epoch + 1) % cfg["selection_every"] == 0 or epoch + 1 == cfg["epochs"]
        metrics = evaluate(model, bank_path, device, cfg["batch_size"], output / "scratch") if eligible else None
        improved = metrics is not None and (best_metric is None or metrics["pooled"]["image_pauc"] > best_metric)
        if improved:
            best_state, best_epoch, best_metrics = cpu_copy(model.state_dict()), epoch + 1, metrics
            best_metric, stale_epochs = metrics["pooled"]["image_pauc"], 0
        elif eligible:
            # Three failed 5-epoch validation checks are 15 training epochs,
            # not 15 checks (75 epochs). Do not increment between checks.
            previous_check = max((row["epoch"] for row in history if row["validation"] is not None), default=0)
            stale_epochs += epoch + 1 - previous_check
        _synchronize(device)
        row = {"epoch": epoch + 1, "train_loss": total_loss / seen, "segmentation_loss": total_seg_loss / seen,
               "classification_loss": total_cls_loss / seen, "images_seen": seen,
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
    validate_config(cfg)
    output = Path(output)
    gate_path = output.parents[1] / "screen-result.json"
    if not gate_path.is_file() or json.loads(gate_path.read_text()) != gate_result:
        raise ValueError("Global gate must be recorded before any calibration read")
    check_identity(declaration)
    validate_binding(declaration, cfg, manifest_path, bank_path, output)
    torch.set_num_threads(cfg["cpu_threads"])
    device = select_device(cfg["device"])
    manifest, bank = load_manifest(manifest_path), load_bank(bank_path, verify_sources=False)
    selected_path = output / "selected.pt"
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    validate_checkpoint(selected, cfg, manifest, bank, declaration, device, selected=True)
    normals = [row for row in manifest["splits"]["calibration"] if row["label"] == 0]
    if not normals or ("inputs" in declaration and len(normals) != declaration["inputs"]["calibration_normal_count"]):
        raise ValueError("Original normal calibration membership differs")
    result_row = gate_result["control"] if cfg["candidate"] == "control" else gate_result["candidate"]
    selected_hash = sha256(selected_path)
    if result_row["checkpoint_sha256"] != selected_hash:
        raise ValueError("Selected weights changed after the global gate")
    identity = {"selected_checkpoint_sha256": selected_hash, "gate_digest": config_digest(gate_result)}
    checkpoint_path = output / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        check_score_identity(checkpoint, cfg)
        if not isinstance(checkpoint.get("threshold"), (int, float)) or not np.isfinite(checkpoint["threshold"]):
            raise ValueError("Cached calibration requires a finite threshold")
        calibration = checkpoint.get("calibration", {})
        scores = np.asarray(calibration.get("scores", []), dtype=np.float64)
        if (scores.ndim != 1 or len(scores) != len(normals) or not np.isfinite(scores).all()
                or calibration.get("normal_count") != len(normals)
                or calibration.get("split") != "calibration" or calibration.get("quantile_method") != "linear"
                or calibration.get("decision_rule") != ">="
                or calibration.get("calibration_exceedances") != int(np.sum(scores >= checkpoint["threshold"]))):
            raise ValueError("Cached calibration scores or membership metadata differ")
        for key in ("declaration_digest", "provenance", "config", "manifest_digest", "best_epoch", "best_metric",
                    "best_validation", "model_selection", "decision_model_config", "parameter_counts"):
            if checkpoint.get(key) != selected.get(key):
                raise ValueError("Cached calibration selection/provenance differs")
        if (checkpoint.get("calibration_identity") != identity or checkpoint.get("config_digest") != config_digest(cfg)
                or checkpoint.get("split_digest") != manifest["split_digest"] or checkpoint.get("bank_digest") != bank["bank_digest"]
                or checkpoint.get("checkpoint_kind") != "inference" or checkpoint.get("threshold") is None):
            raise ValueError("Existing calibrated checkpoint has incompatible identity")
        if (checkpoint["model_state"].keys() != selected["model_state"].keys()
                or any(not torch.equal(checkpoint["model_state"][key], value) for key, value in selected["model_state"].items())
                or not np.isfinite(checkpoint["threshold"])
                or checkpoint["threshold"] != checkpoint["calibration"]["threshold"]
                or checkpoint["calibration"].get("model_kind") != checkpoint["model_kind"]
                or checkpoint["calibration"].get("score_definition") != checkpoint["score_definition"]
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
    verify_records(normals, include_masks=False)  # positive calibration images remain unopened
    model = make_model(cfg).to(device)
    model.load_state_dict(selected["model_state"])
    calibration = calibrate(model, normals, cfg["preprocess"], cfg["batch_size"], device, .9)
    checkpoint = {**selected, "checkpoint_kind": "inference", "threshold": calibration["threshold"],
                  "calibration": calibration, "calibration_identity": identity,
                  "calibration_status": "fit once after immutable two-run validation gate passed",
                  "threshold_source": "separate original clean normal calibration split"}
    atomic_torch_save(checkpoint_path, checkpoint)
    atomic_json(output / "calibration-summary.json", {k: v for k, v in checkpoint.items() if k not in ("model_state", "history")})
    return checkpoint
