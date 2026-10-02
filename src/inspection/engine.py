"""Load a trusted frozen checkpoint once and inspect images consistently."""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .preprocessing import decode_image, normalize_spec, prepare_image, restore_map

JOINT = "joint_reconstruction_segmentation"
RECONSTRUCTION = "normal_only_denoising_reconstruction"


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def model_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        for item in (name.encode(), str(value.dtype).encode(), str(tuple(value.shape)).encode(), value.numpy().tobytes()):
            digest.update(item)
    return digest.hexdigest()


def score_maps(maps, valid_mask=None):
    """Top 1% mean over actual pixels, excluding letterbox padding."""
    if valid_mask is None:
        pixels = maps.flatten(1)
        return pixels.topk(max(1, int(pixels.shape[1] * 0.01)), dim=1).values.mean(dim=1)
    valid_mask = valid_mask.to(maps.device).bool()
    if valid_mask.shape != maps.shape:
        raise ValueError("Valid-pixel mask must match the anomaly map")
    scores = []
    for anomaly_map, valid in zip(maps, valid_mask):
        pixels = anomaly_map[valid]
        if not pixels.numel():
            raise ValueError("Image has no valid pixels")
        scores.append(pixels.topk(max(1, int(pixels.numel() * 0.01))).values.mean())
    return torch.stack(scores)


def render_maps(image, native_map, model_map, display_max):
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError("Display scale must be finite and positive")
    strength = np.clip(native_map / display_max, 0, 1)[..., None] * 0.65
    rgb = np.asarray(image, dtype=np.float32)
    color = np.broadcast_to(np.array([255, 64, 32], dtype=np.float32), rgb.shape)
    overlay = Image.fromarray(np.round(rgb * (1 - strength) + color * strength).astype(np.uint8))
    heatmap = Image.fromarray(np.round(np.clip(model_map / display_max, 0, 1) * 255).astype(np.uint8))
    return heatmap, overlay


class InspectionEngine:
    def __init__(self, checkpoint_path, device="auto"):
        from .train import select_device
        from .model import InspectionModel
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint.get("schema_version", 1)) not in {1, 2}:
            raise ValueError("Unsupported checkpoint schema version")
        config = checkpoint.get("config", {})
        self.category = config.get("category", checkpoint.get("category"))
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("Checkpoint category is required")
        self.threshold = float(checkpoint.get("threshold", float("nan")))
        if not np.isfinite(self.threshold):
            raise ValueError("Checkpoint requires a finite frozen calibration threshold")
        self.image_size = checkpoint.get("image_size", config.get("image_size", 128))
        self.preprocess = normalize_spec(checkpoint.get("preprocess", checkpoint.get("preprocessing")), self.image_size)
        self.model_kind = checkpoint.get("model_kind", JOINT)
        model_config = checkpoint.get("model_config", {"base_channels": config.get("base_channels", 16)})
        if self.model_kind == JOINT:
            self.model = InspectionModel(**model_config)
        elif self.model_kind == RECONSTRUCTION:
            from .baseline import ReconstructionBaseline
            self.model = ReconstructionBaseline(**model_config)
        elif self.model_kind in {"segmentation_only", "supervised_segmentation"}:
            from .model import SegmentationOnlyModel
            self.model = SegmentationOnlyModel(**model_config)
        elif self.model_kind == "patchcore_imagenet":
            reference_size = int(model_config.get("image_size", 256))
            if (self.preprocess["mode"] != "square" or
                    self.preprocess["height"] != reference_size or
                    self.preprocess["width"] != reference_size):
                raise ValueError("PatchCore requires square preprocessing matching its model image_size")
            if str(device) not in {"auto", "cpu"}:
                raise ValueError("The reference PatchCore adapter uses exact CPU nearest-neighbor search")
            from .patchcore_baseline import PatchCoreReference
            device = "cpu"
            self.model = PatchCoreReference(**model_config)
        else:
            raise ValueError(f"Unsupported checkpoint model kind: {self.model_kind}")
        if self.model_kind == "patchcore_imagenet":
            self.model.load_export_state(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])
        self.device = select_device(str(device))
        self.model.to(self.device).eval()
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        self.model_sha256 = model_digest(checkpoint["model_state"])
        self.score_definition = ("mean of highest 1% channel-mean absolute reconstruction errors" if self.model_kind == RECONSTRUCTION
                                 else "mean of highest 1% sigmoid pixel activations") + "; excludes padding; not a calibrated defect probability"
        if self.model_kind == "patchcore_imagenet":
            self.score_definition = checkpoint["score_definition"]
        self.display_max = float(checkpoint.get("display_max", max(2 * self.threshold, 1e-8)))
        if not np.isfinite(self.display_max) or self.display_max <= 0:
            raise ValueError("Checkpoint display scale must be finite and positive")
        self._warmed = False

    @torch.inference_mode()
    def predict_tensor(self, batch, valid_mask=None):
        if batch.ndim != 4 or batch.shape[1] != 3 or not torch.isfinite(batch).all():
            raise ValueError("Expected finite BCHW RGB images")
        if self.model_kind == "patchcore_imagenet":
            maps, scores = self.model.predict_batch(batch.to(self.device))
        else:
            _, prediction = self.model(batch.to(self.device))
            maps = prediction if self.model_kind == RECONSTRUCTION else torch.sigmoid(prediction)
            scores = score_maps(maps, valid_mask)
        if not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
            raise RuntimeError("Model returned non-finite predictions")
        return maps, scores

    def inspect(self, image, category=None):
        if category is not None and category != self.category:
            raise ValueError(f"This checkpoint inspects {self.category}, not {category}")
        started = time.perf_counter()
        original = decode_image(image)
        tensor, valid, transform = prepare_image(original, self.preprocess)
        batch, valid = tensor[None].to(self.device), valid[None].to(self.device)
        if not self._warmed:
            for _ in range(3):
                self.predict_tensor(batch, valid)
            self._warmed = True
        synchronize(self.device)
        inference_started = time.perf_counter()
        maps, scores = self.predict_tensor(batch, valid)
        synchronize(self.device)
        inference_ms = (time.perf_counter() - inference_started) * 1000
        raw_map, score = maps.cpu().numpy()[0, 0], float(scores.cpu()[0])
        native_map = restore_map(raw_map, transform)
        heatmap, overlay = render_maps(original, native_map, raw_map, self.display_max)
        result = {
            "schema_version": 2, "category": self.category, "model_kind": self.model_kind,
            "category_source": "user-selected checkpoint; no automatic category recognition",
            "score": score, "score_definition": self.score_definition, "threshold": self.threshold,
            "decision_rule": "score >= threshold", "decision": "defective" if score >= self.threshold else "good",
            "predicted_defective": score >= self.threshold,
            "checkpoint_sha256": self.checkpoint_sha256, "model_sha256": self.model_sha256,
            "model_image_size": [self.preprocess["width"], self.preprocess["height"]],
            "original_image_size": list(original.size), "preprocess": self.preprocess,
            "device": str(self.device), "warmup_forwards": 3, "inference_ms": inference_ms,
            "inference_scope": "warm model forward and scoring; excludes preprocessing, transfer and rendering",
            "inspection_ms": (time.perf_counter() - started) * 1000,
            "inspection_scope": "decode, prepare, infer, transfer and render; first call includes warmup; excludes model loading",
            "heatmap": "heatmap.png", "overlay": "overlay.png", "raw_map": "anomaly-map.npz",
            "display_scale": {"minimum": 0.0, "maximum": self.display_max,
                              "definition": "fixed checkpoint display range; saturation is not a pixel decision"},
            "heatmap_encoding": "8-bit display only; float32 activations are in anomaly-map.npz",
            "smoke_checkpoint": bool(self.checkpoint.get("smoke_run", False)),
            "limitations": ["Use photographs matching the selected category and dataset imaging conditions.",
                            "Anomaly scores are not probabilities; colored regions are not confirmed defect boundaries.",
                            "Dataset results do not establish reliability on a new production line."],
        }
        return {"summary": result, "image": original, "raw_map": raw_map,
                "native_map": native_map, "heatmap": heatmap, "overlay": overlay}
