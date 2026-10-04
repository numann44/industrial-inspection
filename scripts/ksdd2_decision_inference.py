"""Isolated inference for trusted, calibrated decision-screen checkpoints.

The shared InspectionEngine is unchanged. This adapter checks embedded training,
selection and calibration identity; the evaluation coordinator additionally pins
the external declaration, gate and checkpoint SHA before permitting test access.
Classification logits are not probabilities. Maps always show sigmoid pixel
segmentation activations on a fixed 0..1 display scale, independent of score units.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import re
import time

import numpy as np
from PIL import ImageEnhance, ImageFilter
import torch

from experiments.ksdd2_decision.model import check_score_identity, make_model, parameter_counts
from experiments.ksdd2_decision.protocol import PROTOCOL, SELECTION_RULE, source_hashes
from experiments.ksdd2_decision.trainer import validate_config
from inspection.checkpointing import config_digest
from inspection.engine import InspectionEngine, model_digest, synchronize
from inspection.evaluate import compute_metrics
from inspection.ksdd2 import load_manifest
from inspection.preprocessing import decode_image, normalize_spec, prepare_image
from inspection.reporting import checkpoint_provenance
from inspection.train import select_device


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _validate_checkpoint(checkpoint):
    if (checkpoint.get("schema_version") != 2 or checkpoint.get("checkpoint_kind") != "inference"
            or checkpoint.get("resumable") is not False):
        raise ValueError("Expected a calibrated, nonresumable decision-screen inference checkpoint")
    cfg = checkpoint.get("config", {})
    try:
        validate_config(cfg)
        check_score_identity(checkpoint, cfg)
    except (KeyError, TypeError) as error:
        raise ValueError("Missing decision-screen configuration/model/score identity") from error
    if (cfg["protocol"] != PROTOCOL or checkpoint.get("config_digest") != config_digest(cfg)
            or checkpoint.get("model_config") != {"base_channels": cfg["base_channels"]}):
        raise ValueError("Decision-screen configuration digest or architecture differs")
    if checkpoint.get("category") != "kolektor_surface" or cfg.get("category", "kolektor_surface") != "kolektor_surface":
        raise ValueError("Decision checkpoint category must be kolektor_surface")
    preprocess = normalize_spec(checkpoint.get("preprocess"))
    if preprocess != cfg["preprocess"] or preprocess["mode"] != "letterbox":
        raise ValueError("Checkpoint preprocessing differs from declared letterbox geometry")
    for key in ("declaration_digest", "manifest_digest", "split_digest", "bank_digest"):
        if not _digest(checkpoint.get(key)):
            raise ValueError(f"Invalid frozen {key}")
    provenance = checkpoint.get("provenance", {})
    if (provenance.get("decision_source_files") != source_hashes()
            or any(provenance.get(key) != checkpoint[key] for key in ("config_digest", "split_digest", "bank_digest"))):
        raise ValueError("Frozen decision source/config/split provenance differs")
    if (checkpoint.get("model_selection") != SELECTION_RULE
            or checkpoint.get("training_kind") != "random_initialization_real_defect_supervision_detached_decision_screen"
            or checkpoint.get("test_status") != "development-inspected/exploratory for all reused KSDD2 tests"):
        raise ValueError("Checkpoint is not from the declared exploratory decision screen")
    best_epoch, completed = checkpoint.get("best_epoch"), checkpoint.get("completed_epochs")
    if (not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or not isinstance(completed, int) or isinstance(completed, bool)
            or not 1 <= best_epoch <= completed <= cfg["epochs"]
            or (best_epoch % cfg["selection_every"] != 0 and best_epoch != cfg["epochs"])):
        raise ValueError("Invalid validation-selected epoch metadata")
    validation = checkpoint.get("best_validation", {})
    best_metric = checkpoint.get("best_metric")
    if (not isinstance(best_metric, (int, float)) or isinstance(best_metric, bool) or not np.isfinite(best_metric) or not 0 <= best_metric <= 1
            or validation.get("pooled", {}).get("image_pauc") != best_metric
            or validation.get("bank_digest") != checkpoint["bank_digest"]
            or validation.get("split_digest") != checkpoint["split_digest"]
            or validation.get("model_kind") != checkpoint["model_kind"]
            or validation.get("score_definition") != checkpoint["score_definition"]
            or validation.get("selection_rule") != SELECTION_RULE):
        raise ValueError("Validation selection metadata differs from checkpoint identity")
    threshold = checkpoint.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not np.isfinite(threshold):
        raise ValueError("A finite frozen calibration threshold is required")
    calibration = checkpoint.get("calibration", {})
    try:
        scores = np.asarray(calibration.get("scores", []), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid saved calibration scores") from error
    if (scores.ndim != 1 or not scores.size or not np.isfinite(scores).all()
            or calibration.get("normal_count") != len(scores) or calibration.get("split") != "calibration"
            or calibration.get("quantile") != .9 or calibration.get("quantile_method") != "linear"
            or calibration.get("decision_rule") != ">=" or calibration.get("threshold") != threshold
            or threshold != float(np.quantile(scores, .9, method="linear"))
            or calibration.get("calibration_exceedances") != int(np.sum(scores >= threshold))
            or calibration.get("model_kind") != checkpoint["model_kind"]
            or calibration.get("score_definition") != checkpoint["score_definition"]):
        raise ValueError("Calibration threshold/scores/score identity differ")
    if cfg["candidate"] == "control" and (np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("Control segmentation scores must lie within 0..1")
    calibration_identity = checkpoint.get("calibration_identity", {})
    if (not _digest(calibration_identity.get("selected_checkpoint_sha256"))
            or not _digest(calibration_identity.get("gate_digest"))
            or checkpoint.get("calibration_status") != "fit once after immutable two-run validation gate passed"
            or checkpoint.get("threshold_source") != "separate original clean normal calibration split"):
        raise ValueError("Calibration lacks the frozen selected-checkpoint/gate identity")
    return cfg, preprocess


class DecisionInspectionEngine(InspectionEngine):
    """Shared inspection/geometry contract with an explicitly different scorer."""
    def __init__(self, checkpoint_path, device="mps"):
        self.checkpoint_path = Path(checkpoint_path)
        payload = self.checkpoint_path.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        self.checkpoint = checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        cfg, self.preprocess = _validate_checkpoint(checkpoint)
        self.category = checkpoint["category"]
        self.candidate = cfg["candidate"]
        self.threshold = float(checkpoint["threshold"])
        self.model_kind = checkpoint["model_kind"]
        self.score_definition = checkpoint["score_definition"]
        self.model = make_model(cfg)
        if parameter_counts(self.model) != checkpoint.get("parameter_counts"):
            raise ValueError("Checkpoint parameter counts differ from the declared architecture")
        state = checkpoint.get("model_state", {})
        if not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Checkpoint weights must be finite tensors")
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("Checkpoint weights do not match the declared decision model") from error
        self.device = select_device(str(device))
        self.model.to(self.device).eval()
        self.model_sha256 = model_digest(state)
        self.display_max = 1.0  # Pixel activation scale; unrelated to negative/positive image logits.
        self.image_size = [self.preprocess["height"], self.preprocess["width"]]
        self._warmed = False

    @torch.inference_mode()
    def predict_tensor(self, batch, valid_mask=None):
        if (not isinstance(batch, torch.Tensor) or batch.ndim != 4 or batch.shape[0] < 1
                or batch.shape[1] != 3 or batch.dtype != torch.float32
                or tuple(batch.shape[-2:]) != (self.preprocess["height"], self.preprocess["width"])
                or not torch.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any()):
            raise ValueError("Expected finite float32 BCHW RGB images in [0,1] at declared model resolution")
        if (not isinstance(valid_mask, torch.Tensor) or valid_mask.shape != batch[:, :1].shape
                or not torch.isfinite(valid_mask).all() or not ((valid_mask == 0) | (valid_mask == 1)).all()
                or not valid_mask.bool().flatten(1).any(dim=1).all()):
            raise ValueError("Every letterboxed image requires an aligned nonempty binary valid mask")
        logits, scores = self.model(batch.to(self.device), valid_mask.to(self.device).bool())
        maps = torch.sigmoid(logits)
        if (maps.shape != batch[:, :1].shape or scores.shape != (len(batch),)
                or not torch.isfinite(maps).all() or not torch.isfinite(scores).all()):
            raise RuntimeError("Decision model returned invalid map/score values")
        return maps, scores

    def inspect(self, image, category=None):
        result = super().inspect(image, category)
        result["summary"].update({"candidate": self.candidate, "experimental_status": "exploratory",
            "map_definition": "sigmoid segmentation pixel activations; not the image classification score or probability",
            "declaration_digest": self.checkpoint["declaration_digest"],
            "calibration_identity": self.checkpoint["calibration_identity"]})
        result["summary"]["limitations"].append(
            "Exploratory reused-test method; classification-head gradients do not train the segmentation backbone.")
        return result


def evaluation_context(manifest_path, checkpoint_path, device="mps", batch_size=8):
    """Reject incompatible checkpoint/manifest identities before any test bytes."""
    _positive_integer(batch_size, "batch_size")
    engine = DecisionInspectionEngine(checkpoint_path, device)
    manifest = load_manifest(manifest_path)
    records = manifest["splits"]["test"]
    if (manifest.get("dataset") != "KolektorSDD2" or not records
            or manifest["split_digest"] != engine.checkpoint["split_digest"]
            or any(row["category"] != engine.category for row in records)):
        raise ValueError("Decision checkpoint and nonempty audited category/test split differ")
    if sum(not row["label"] for row in manifest["splits"]["calibration"]) != engine.checkpoint["calibration"]["normal_count"]:
        raise ValueError("Checkpoint calibration count differs from the audited manifest")
    load_manifest(manifest_path, verify_splits=("test",))
    return manifest, engine


def additional_sources():
    root = Path(__file__).resolve().parents[1]
    result = source_hashes()
    for relative in ("scripts/ksdd2_decision_inference.py", "scripts/ksdd2_decision_native.py", "scripts/ksdd2_native_stream.py"):
        path = root / relative
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def stress_test_checkpoint(manifest_path, checkpoint_path, device="mps", batch_size=8):
    manifest, engine = evaluation_context(manifest_path, checkpoint_path, device, batch_size)
    records = manifest["splits"]["test"]

    def jpeg(image):
        with io.BytesIO() as stream:
            image.save(stream, format="JPEG", quality=60)
            return decode_image(stream.getvalue())

    conditions = {"original": lambda image: image,
                  "gaussian_blur_radius_1": lambda image: image.filter(ImageFilter.GaussianBlur(1)),
                  "brightness_0.8": lambda image: ImageEnhance.Brightness(image).enhance(.8),
                  "brightness_1.2": lambda image: ImageEnhance.Brightness(image).enhance(1.2),
                  "jpeg_quality_60": jpeg}
    labels = [record["label"] for record in records]
    results, reference_scores = {}, None
    for name, perturb in conditions.items():
        scores = []
        for start in range(0, len(records), batch_size):
            prepared = [prepare_image(perturb(decode_image(row["image"])), engine.preprocess)
                        for row in records[start:start + batch_size]]
            _, values = engine.predict_tensor(torch.stack([p[0] for p in prepared]), torch.stack([p[1] for p in prepared]))
            scores.extend(values.cpu().tolist())
        empty = np.zeros((len(labels), 1, 1, 1), dtype=np.float32)
        metrics = compute_metrics(labels, scores, empty, empty, engine.threshold, include_pixel_metric=False)
        metrics.pop("evaluation_resolution"); metrics.pop("pixel_average_precision")
        if reference_scores is None:
            reference_scores = np.asarray(scores)
        original_decisions = reference_scores >= engine.threshold
        decisions = np.asarray(scores) >= engine.threshold
        metrics.update({"decision_flips": int(np.sum(original_decisions != decisions)),
                        "accepted_to_flagged": int(np.sum(~original_decisions & decisions)),
                        "flagged_to_accepted": int(np.sum(original_decisions & ~decisions)),
                        "mean_absolute_score_change": float(np.mean(np.abs(reference_scores - scores))),
                        "predictions": [{"image": row["image"], "label": row["label"], "score": float(score),
                                         "predicted_defective": bool(score >= engine.threshold)}
                                        for row, score in zip(records, scores)]})
        results[name] = metrics
    result = {"status": "exploratory paired dataset stress test", "experimental_status": "exploratory",
              "category": engine.category, "candidate": engine.candidate, "model_kind": engine.model_kind,
              "split_digest": manifest["split_digest"], "declaration_digest": engine.checkpoint["declaration_digest"],
              "checkpoint_sha256": engine.checkpoint_sha256, "threshold": engine.threshold,
              "score_definition": engine.score_definition,
              "threshold_source": "frozen checkpoint calibration; unchanged for every perturbation",
              "results": results, "pixel_metrics": "not computed: image decision stress test only",
              "limitations": "Reused-test exploratory evidence; perturbations do not establish reliability under unseen acquisition conditions."}
    result.update(checkpoint_provenance(checkpoint_path))
    if result["checkpoint_sha256"] != engine.checkpoint_sha256:
        raise ValueError("Checkpoint changed during stress evaluation")
    result["evaluation_additional_source_sha256"] = additional_sources()
    return result


def benchmark_checkpoint(checkpoint_path, image_path, device="cpu", iterations=32, warmup=3):
    _positive_integer(iterations, "iterations"); _positive_integer(warmup, "warmup")
    engine = DecisionInspectionEngine(checkpoint_path, device)
    tensor, valid, _ = prepare_image(decode_image(image_path), engine.preprocess)
    batch, valid = tensor[None].to(engine.device), valid[None].to(engine.device)
    for _ in range(warmup):
        engine.predict_tensor(batch, valid)
    synchronize(engine.device)
    timings = []
    for _ in range(iterations):
        synchronize(engine.device)
        start = time.perf_counter()
        engine.predict_tensor(batch, valid)
        synchronize(engine.device)
        timings.append((time.perf_counter() - start) * 1000)
    return {"device": str(engine.device), "candidate": engine.candidate, "model_kind": engine.model_kind,
            "score_definition": engine.score_definition, "checkpoint_sha256": engine.checkpoint_sha256,
            "model_sha256": engine.model_sha256, "preprocess": engine.preprocess,
            "measurements": iterations, "warmup_forwards": warmup,
            "single_image_median_ms": float(np.median(timings)), "single_image_p95_ms": float(np.quantile(timings, .95)),
            "scope": "warm forward and scoring only; excludes model loading, decode, preprocessing, transfer and rendering",
            "benchmark_condition": "Run after training and other inference jobs finish; this function cannot enforce system idleness.",
            "cpu_threads": torch.get_num_threads(), "experimental_status": "exploratory"}
