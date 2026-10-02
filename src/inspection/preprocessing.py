"""Shared image geometry for training, evaluation and interactive inspection."""
from __future__ import annotations
import io
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(source) -> Image.Image:
    """Decode oriented RGB; uploaded bytes are never written to disk."""
    if isinstance(source, bytes):
        if len(source) > MAX_UPLOAD_BYTES:
            raise ValueError("Image exceeds the 10 MiB upload limit")
        source = io.BytesIO(source)
    try:
        image = source.copy() if isinstance(source, Image.Image) else Image.open(source)
        with image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 20 megapixel limit")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot decode this image; upload a valid PNG, JPEG or WebP") from exc


def normalize_spec(spec: dict | None, image_size=128) -> dict:
    spec = dict(spec or {})
    height, width = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
    mode = spec.get("mode", "square")
    height, width = int(spec.get("height", height)), int(spec.get("width", width))
    if mode not in {"square", "letterbox"} or min(height, width) < 16 or max(height, width) > 2048:
        raise ValueError("Expected square or letterbox preprocessing with dimensions 16..2048")
    if mode == "square" and height != width:
        raise ValueError("Square preprocessing requires equal dimensions")
    return {"mode": mode, "height": height, "width": width}


def _geometry(image, spec):
    spec = normalize_spec(spec)
    width, height = spec["width"], spec["height"]
    if spec["mode"] == "letterbox":
        scale = min(width / image.width, height / image.height)
        resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    else:
        resized = (width, height)
    return {**spec, "original_size": list(image.size), "resized_size": list(resized),
            "left": (width - resized[0]) // 2, "top": (height - resized[1]) // 2}


def prepare_image(image: Image.Image, spec: dict):
    transform = _geometry(image, spec)
    resized = image.convert("RGB").resize(tuple(transform["resized_size"]), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (transform["width"], transform["height"]), 0)
    left, top = transform["left"], transform["top"]
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32).copy() / 255
    valid = torch.zeros((1, transform["height"], transform["width"]), dtype=torch.bool)
    valid[:, top:top + resized.height, left:left + resized.width] = True
    return torch.from_numpy(array).permute(2, 0, 1), valid, transform


def prepare_mask(mask: Image.Image, spec: dict):
    transform = _geometry(mask, spec)
    resized = mask.convert("L").resize(tuple(transform["resized_size"]), Image.Resampling.NEAREST)
    canvas = Image.new("L", (transform["width"], transform["height"]), 0)
    canvas.paste(resized, (transform["left"], transform["top"]))
    return torch.from_numpy((np.asarray(canvas).copy() > 0).astype(np.float32))[None]


def restore_map(anomaly_map, transform):
    array = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite 2D anomaly map")
    model_map = Image.fromarray(array).resize((transform["width"], transform["height"]), Image.Resampling.BILINEAR)
    left, top = transform["left"], transform["top"]
    width, height = transform["resized_size"]
    return np.asarray(model_map.crop((left, top, left + width, top + height)).resize(
        tuple(transform["original_size"]), Image.Resampling.BILINEAR), dtype=np.float32).copy()
