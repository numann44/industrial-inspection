"""Procedural synthetic defects using only the current clean training image.

No external texture dataset or real test defects are used. These corruptions are
training signals; their similarity to real manufacturing defects must be tested.
"""

import torch


def _random(generator, low=0.0, high=1.0):
    return low + (high - low) * torch.rand((), generator=generator).item()


def synthesize(image, generator, normal_probability=0.2, mode=None):
    """Return (corrupted RGB CHW image, binary 1HW mask) on CPU."""
    if image.device.type != "cpu":
        raise ValueError("Synthetic dataset augmentation expects CPU tensors")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB image with shape (3, H, W)")
    height, width = image.shape[-2:]
    mask = torch.zeros((1, height, width), dtype=image.dtype)
    if mode is None and _random(generator) < normal_probability:
        return image.clone(), mask
    if mode is None:
        mode = ("scratch", "appearance", "texture")[
            int(torch.randint(3, (), generator=generator).item())
        ]
    if mode not in ("scratch", "appearance", "texture"):
        raise ValueError(f"Unknown corruption mode: {mode}")
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij"
    )
    if mode == "scratch":
        x0, y0, x1, y1 = [_random(generator, 0.1, 0.9) for _ in range(4)]
        dx, dy = x1 - x0, y1 - y0
        t = ((xx - x0) * dx + (yy - y0) * dy) / max(dx * dx + dy * dy, 1e-6)
        t = t.clamp(0, 1)
        distance = ((xx - x0 - t * dx) ** 2 + (yy - y0 - t * dy) ** 2).sqrt()
        region = distance < _random(generator, 1.0 / min(height, width), 0.035)
        color = torch.rand((3, 1, 1), generator=generator)
        replacement = color.expand_as(image)
    else:
        cx, cy = _random(generator, 0.15, 0.85), _random(generator, 0.15, 0.85)
        rx, ry = _random(generator, 0.06, 0.25), _random(generator, 0.06, 0.25)
        angle = torch.atan2((yy - cy) / ry, (xx - cx) / rx)
        irregular_radius = 1 + 0.2 * torch.sin(5 * angle + _random(generator, 0, 6.28))
        region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < irregular_radius**2
        if mode == "texture":
            shifts = (int(_random(generator, height / 5, height / 2)),
                      int(_random(generator, width / 5, width / 2)))
            replacement = torch.roll(image, shifts=shifts, dims=(1, 2))
            replacement = replacement + torch.randn(image.shape, generator=generator) * 0.08
        else:
            color = torch.rand((3, 1, 1), generator=generator)
            replacement = 0.25 * image + 0.75 * color
    mask[0] = region.to(image.dtype)
    # Hard support keeps the generated mask exactly aligned with altered pixels.
    strength = _random(generator, 0.55, 1.0)
    corrupted = image * (1 - strength * mask) + replacement.clamp(0, 1) * strength * mask
    return corrupted.clamp(0, 1), mask
