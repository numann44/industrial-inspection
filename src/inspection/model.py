"""Compact reconstruction and segmentation networks trained from random weights."""

import torch
from torch import nn
from torch.nn import functional as F

JOINT_MODEL_KIND = "joint_reconstruction_segmentation"
SEGMENTATION_MODEL_KIND = "segmentation_only"
SCORE_DEFINITION = "mean of highest 1% sigmoid anomaly-map pixels; not a calibrated probability"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, image):
        return self.layers(image)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int):
        super().__init__()
        widths = [base_channels * (2**i) for i in range(4)]
        self.encoders = nn.ModuleList(
            [ConvBlock(in_channels, widths[0])]
            + [ConvBlock(widths[i - 1], widths[i]) for i in range(1, 4)]
        )
        self.decoders = nn.ModuleList(
            [ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in (2, 1, 0)]
        )
        self.output = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, image):
        skips = []
        x = image
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.avg_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.output(x)


class InspectionModel(nn.Module):
    """A lightweight DRAEM-inspired variant, not a reproduction of its architecture."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.model_kind = JOINT_MODEL_KIND
        self.reconstruction = SmallUNet(3, 3, base_channels)
        self.segmentation = SmallUNet(6, 1, base_channels)

    def forward(self, image):
        reconstruction = torch.sigmoid(self.reconstruction(image))
        logits = self.segmentation(torch.cat((image, reconstruction), dim=1))
        return reconstruction, logits


class SegmentationOnlyModel(nn.Module):
    """A single U-Net ablation with the same segmentation target and score rule."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        self.model_kind = SEGMENTATION_MODEL_KIND
        self.segmentation = SmallUNet(3, 1, base_channels)

    def forward(self, image):
        return image, self.segmentation(image)


def create_model(model_kind, model_config):
    if model_kind == JOINT_MODEL_KIND:
        return InspectionModel(**model_config)
    if model_kind in (SEGMENTATION_MODEL_KIND, "supervised_segmentation"):
        return SegmentationOnlyModel(**model_config)
    if model_kind == "normal_only_denoising_reconstruction":
        from inspection.baseline import ReconstructionBaseline
        return ReconstructionBaseline(**model_config)
    raise ValueError(f"Unsupported model kind: {model_kind}")


def anomaly_score(logits: torch.Tensor) -> torch.Tensor:
    """Mean of the highest 1% of pixel activations; not a defect probability."""
    pixels = torch.sigmoid(logits).flatten(1)
    count = max(1, int(pixels.shape[1] * 0.01))
    return pixels.topk(count, dim=1).values.mean(dim=1)
