"""Procedural synthetic defects using only the current clean training image.

No external texture dataset or real test defects are used. These corruptions are
training signals; their similarity to real manufacturing defects must be tested.
"""

import torch
from torch.nn import functional as F


def _random(generator, low=0.0, high=1.0):
    return low + (high - low) * torch.rand((), generator=generator).item()


def foreground_mask(image):
    """Heuristic metal surface mask from border RGB; never fill the central hole.

    Assumes the photographed part is centered on a relatively uniform background.
    This category-specific heuristic is not a learned/general-purpose segmenter.
    """
    height, width = image.shape[-2:]
    border_width = max(1, min(height, width) // 32)
    border = torch.cat((
        image[:, :border_width].flatten(1), image[:, -border_width:].flatten(1),
        image[:, :, :border_width].flatten(1), image[:, :, -border_width:].flatten(1),
    ), dim=1)
    background = border.median(dim=1).values
    border_distance = (border - background[:, None]).square().mean(dim=0).sqrt()
    median = border_distance.median()
    mad = (border_distance - median).abs().median()
    threshold = max(0.04, float(median + 6 * mad), float(torch.quantile(border_distance, 0.98)))
    distance = (image - background[:, None, None]).square().mean(dim=0).sqrt()
    surface = (distance > threshold).float()[None, None]
    # Opening removes isolated background noise while leaving dark holes open.
    eroded = 1 - F.max_pool2d(1 - surface, 3, stride=1, padding=1)
    opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
    return opened[0].bool()


def _surface_corruption(image, generator, mode, restrict_foreground=True):
    height, width = image.shape[-2:]
    surface = foreground_mask(image)
    interior = (1 - F.max_pool2d(1 - surface.float()[None], 3, stride=1, padding=1))[0].bool()
    allowed = surface if mode == "warp" else interior
    locations = allowed[0].nonzero()
    if not len(locations):
        return image.clone(), torch.zeros_like(surface, dtype=image.dtype)
    if mode == "warp":
        deep_interior = (1 - F.max_pool2d(1 - surface.float()[None], 9, stride=1, padding=4))[0].bool()
        edge_locations = (surface & ~deep_interior)[0].nonzero()
        if len(edge_locations):
            locations = edge_locations
    location = locations[int(torch.randint(len(locations), (), generator=generator))]
    cy, cx = float(location[0]), float(location[1])
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    yy, xx = yy.float(), xx.float()
    if mode == "scratch":
        angle = _random(generator, 0, 6.283185)
        length = _random(generator, 0.08, 0.35) * min(height, width)
        direction = torch.tensor(angle)
        dx, dy = float(torch.cos(direction)), float(torch.sin(direction))
        along = (xx - cx) * dx + (yy - cy) * dy
        across = ((xx - cx) * dy - (yy - cy) * dx).abs()
        # Full scratch width is 1--3 pixels at 256px, rather than a thick colored stripe.
        half_width = _random(generator, 0.5, 1.5) * min(height, width) / 256
        region = (along.abs() < length / 2) & (across < max(0.5, half_width))
        luminance = image.mean(dim=0, keepdim=True)
        contrast = _random(generator, 0.12, 0.38) * (-1 if _random(generator) < 0.5 else 1)
        replacement = (luminance + contrast).expand_as(image)
    else:
        rx = _random(generator, 0.025, 0.12) * width
        ry = _random(generator, 0.025, 0.12) * height
        angle = torch.atan2((yy - cy) / ry, (xx - cx) / rx)
        radius = 1 + 0.15 * torch.sin(5 * angle + _random(generator, 0, 6.28))
        region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < radius**2
        if mode == "lighting":
            replacement = image * _random(generator, 0.6, 1.4) + _random(generator, -0.04, 0.04)
        elif mode == "appearance":
            color = torch.rand((3, 1, 1), generator=generator)
            replacement = 0.3 * image + 0.7 * color
        elif mode == "texture":
            shifts = (int(_random(generator, 0.02, 0.08) * height),
                      int(_random(generator, 0.02, 0.08) * width))
            replacement = torch.roll(image, shifts=shifts, dims=(1, 2))
            # Both destination and copied source must be on the original surface.
            if restrict_foreground:
                allowed = allowed & torch.roll(surface, shifts=shifts, dims=(1, 2))
            replacement = replacement + torch.randn(image.shape, generator=generator) * 0.015
        elif mode == "warp":
            # Copy a slightly shifted local neighborhood from this same image.
            # At an edge this can remove material inward; outward deformations
            # cannot be represented by strict original-foreground support.
            dx = _random(generator, -0.035, 0.035) * width
            dy = _random(generator, -0.035, 0.035) * height
            grid = torch.stack((2 * (xx + dx) / (width - 1) - 1,
                                2 * (yy + dy) / (height - 1) - 1), dim=-1)[None]
            replacement = F.grid_sample(image[None], grid, mode="bilinear",
                                        padding_mode="border", align_corners=True)[0]
        else:
            raise ValueError(f"Unknown foreground corruption mode: {mode}")
    # The ablation changes support only; random draws, centers and families are
    # identical between restricted and unrestricted paired samples.
    support = region[None] & allowed if restrict_foreground else region[None]
    strength = _random(generator, 0.7, 1.0)
    replacement = image * (1 - strength) + replacement.clamp(0, 1) * strength
    changed = (replacement - image).abs().amax(dim=0, keepdim=True) > 1e-6
    mask = (support & changed).to(image.dtype)
    corrupted = torch.where(mask.bool().expand_as(image), replacement, image)
    return corrupted.clamp(0, 1), mask


def synthesize(image, generator, normal_probability=0.2, mode=None, strategy="legacy",
               restrict_foreground=True, scratch_enabled=True):
    """Return (corrupted RGB CHW image, binary 1HW mask) on CPU."""
    if image.device.type != "cpu":
        raise ValueError("Synthetic dataset augmentation expects CPU tensors")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected an RGB image with shape (3, H, W)")
    height, width = image.shape[-2:]
    mask = torch.zeros((1, height, width), dtype=image.dtype)
    if mode is None and _random(generator) < normal_probability:
        return image.clone(), mask
    if strategy == "foreground":
        if mode is None:
            modes = ("scratch", "scratch", "scratch", "lighting", "texture", "warp", "warp", "appearance")
            if not scratch_enabled:
                modes = tuple(name for name in modes if name != "scratch")
            mode = modes[int(torch.randint(len(modes), (), generator=generator))]
        if mode == "scratch" and not scratch_enabled:
            raise ValueError("Scratch mode is disabled by this configuration")
        return _surface_corruption(image, generator, mode, restrict_foreground)
    if strategy != "legacy":
        raise ValueError(f"Unknown synthesis strategy: {strategy}")
    if mode is None:
        modes = ("scratch", "appearance", "texture") if scratch_enabled else ("appearance", "texture")
        mode = modes[
            int(torch.randint(len(modes), (), generator=generator).item())
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
