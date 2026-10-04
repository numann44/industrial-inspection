"""Deterministic CPU corruptions from normal images; no real defect inputs."""
import torch
from torch.nn import functional as F

from inspection.synthesis import foreground_mask, synthesize as legacy_synthesize

MODES = ("scratch", "scratch", "scratch", "lighting", "texture", "warp", "warp", "appearance")


def uniform(generator, low, high):
    return low + (high - low) * torch.rand((), generator=generator).item()


def local_support(image, generator, *, boundary=False):
    """An irregular compact support, with a one-pixel taper at the edge."""
    height, width = image.shape[-2:]
    surface = foreground_mask(image)
    allowed = surface
    if boundary:
        dilated = F.max_pool2d(surface.float()[None], 5, 1, 2)[0].bool()
        eroded = (1 - F.max_pool2d(1 - surface.float()[None], 5, 1, 2))[0].bool()
        allowed = dilated & ~eroded
    points = allowed[0].nonzero()
    # No silent whole-image fallback when the documented heuristic fails.
    if len(points) == 0:
        return torch.zeros_like(image[:1])
    cy, cx = points[int(torch.randint(len(points), (), generator=generator))].float()
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    rx, ry = uniform(generator, .025, .12) * width, uniform(generator, .025, .12) * height
    dx, dy = (xx - cx) / rx, (yy - cy) / ry
    phase = uniform(generator, 0, 6.283185)
    radius = 1 + .15 * torch.sin(5 * torch.atan2(dy, dx) + phase)
    distance = (dx.square() + dy.square()).sqrt() / radius
    feather = ((1 - distance) * min(rx, ry)).clamp(0, 1)[None]
    # Deformations may cross the original silhouette; appearances remain on it.
    return feather if boundary else feather * surface


def subtle_patch(image, donor, generator):
    if donor.shape != image.shape:
        raise ValueError("The clean donor must have the same CHW shape")
    alpha = local_support(image, generator) * uniform(generator, .15, .75)
    height, width = image.shape[-2:]
    shifts = (int(uniform(generator, -.25, .25) * height),
              int(uniform(generator, -.25, .25) * width))
    replacement = torch.roll(donor, shifts, (1, 2))
    # Donor texture is shifted to local destination mean brightness; its texture
    # variation remains. This package is not an isolated mask-vs-contrast study.
    selected = alpha[0] > 0
    if selected.any():
        offset = image[:, selected].mean(1) - replacement[:, selected].mean(1)
        replacement = (replacement + offset[:, None, None]).clamp(0, 1)
    return image * (1 - alpha) + replacement * alpha


def boundary_warp(image, generator):
    height, width = image.shape[-2:]
    support = local_support(image, generator, boundary=True)
    dx, dy = uniform(generator, -.035, .035) * width, uniform(generator, -.035, .035) * height
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid = torch.stack((2 * (xx + dx * support[0]) / (width - 1) - 1,
                        2 * (yy + dy * support[0]) / (height - 1) - 1), -1)[None]
    moved = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border",
                         align_corners=True)[0]
    # Identity grid interpolation has roundoff outside support; preserve exact
    # originals there so the binary annotation records actual changed pixels.
    return torch.where(support.bool().expand_as(image), moved, image)


def withheld_translation(image, donor, generator):
    """Bank-only operator: a hard rectangular normal patch insertion.

    This operator is never called by the training dispatcher. It is still a
    synthetic proxy, not an independently acquired physical defect.
    """
    height, width = image.shape[-2:]
    ph, pw = max(2, height // 12), max(2, width // 12)
    y = int(torch.randint(height - ph + 1, (), generator=generator))
    x = int(torch.randint(width - pw + 1, (), generator=generator))
    sy = int(torch.randint(height - ph + 1, (), generator=generator))
    sx = int(torch.randint(width - pw + 1, (), generator=generator))
    changed = image.clone()
    changed[:, y:y + ph, x:x + pw] = donor[:, sy:sy + ph, sx:sx + pw]
    return changed


def changed_mask(clean, changed):
    return ((clean - changed).abs().amax(0, keepdim=True) > 1e-6).to(clean.dtype)


def synthesize(image, donor, generator, candidate, normal_probability=.25, mode=None):
    if candidate not in ("control", "A", "B"):
        raise ValueError("Only preregistered control, A and B are supported")
    if image.device.type != "cpu" or donor.device.type != "cpu":
        raise ValueError("Synthesis expects CPU tensors")
    if mode is None:
        if uniform(generator, 0, 1) < normal_probability:
            return image.clone(), torch.zeros_like(image[:1])
        mode = MODES[int(torch.randint(len(MODES), (), generator=generator))]
    if candidate == "control":
        return legacy_synthesize(image, generator, mode=mode, strategy="foreground")
    if mode in ("appearance", "texture"):
        changed = subtle_patch(image, donor, generator)
    elif mode == "warp" and candidate == "B":
        changed = boundary_warp(image, generator)
    elif mode in MODES:
        return legacy_synthesize(image, generator, mode=mode, strategy="foreground")
    else:
        raise ValueError("Unknown training corruption; bank-only operators are forbidden")
    mask = changed_mask(image, changed)
    return torch.where(mask.bool().expand_as(image), changed.clamp(0, 1), image), mask
