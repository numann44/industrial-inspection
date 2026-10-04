"""Segmentation-conditioned classification with completely detached inputs."""
import torch
from torch import nn
from torch.nn import functional as F

from inspection.model import SmallUNet
from inspection.supervised import valid_anomaly_score

CONTROL_KIND = "supervised_segmentation"
HEAD_KIND = "supervised_segmentation_detached_decision_v1"
CONTROL_SCORE = "mean highest 1% sigmoid segmentation pixels within valid letterbox area; not a calibrated probability"
HEAD_SCORE = "raw detached-context classification logit; greater means more defective; not a calibrated probability"


def identity(candidate):
    if candidate == "control":
        return CONTROL_KIND, CONTROL_SCORE
    if candidate == "decision_head":
        return HEAD_KIND, HEAD_SCORE
    raise ValueError("Unknown decision candidate")


class ContextHead(nn.Module):
    def __init__(self, bottleneck_channels):
        super().__init__()
        widths = (bottleneck_channels + 1, 16, 16, 32)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(a, b, 3, padding=1),
                                                nn.GroupNorm(4, b), nn.SiLU())
                                    for a, b in zip(widths[:-1], widths[1:]))
        self.output = nn.Linear(66, 1)

    def forward(self, features, logits, valid):
        # Both paths, including the map-statistics shortcut, are detached.
        features, logits = features.detach(), logits.detach()
        if valid.shape != logits.shape or not valid.bool().flatten(1).any(dim=1).all():
            raise ValueError("Aligned nonempty valid masks required by decision head")
        valid = valid.bool()
        coarse_valid = F.interpolate(valid.float(), size=features.shape[-2:], mode="nearest").bool()
        coarse_logits = F.adaptive_avg_pool2d(logits.masked_fill(~valid, 0), features.shape[-2:])
        coverage = F.adaptive_avg_pool2d(valid.float(), features.shape[-2:])
        coarse_logits = coarse_logits / coverage.clamp_min(1e-8)
        decisions = []
        for index in range(len(features)):
            mask = coarse_valid[index, 0]
            if not mask.any():
                raise ValueError("Image has no valid bottleneck positions")
            positions = torch.nonzero(mask, as_tuple=False)
            top, left = positions.min(dim=0).values.tolist()
            bottom, right = (positions.max(dim=0).values + 1).tolist()
            active = mask[None, None, top:bottom, left:right]
            x = torch.cat((features[index:index + 1], coarse_logits[index:index + 1]), dim=1)
            x = x[:, :, top:bottom, left:right].masked_fill(~active, 0)
            for block in self.blocks:
                # Invalid positions cannot win a max or enter a mean. Re-mask
                # between learned blocks; crop removes letterbox-only borders.
                x = F.max_pool2d(x.masked_fill(~active, -torch.inf), 2, ceil_mode=True)
                active = F.max_pool2d(active.float(), 2, ceil_mode=True).bool()
                x = block(x.masked_fill(~active, 0)).masked_fill(~active, 0)
            count = active.sum().clamp_min(1)
            mean = x.sum(dim=(-1, -2)) / count
            maximum = x.masked_fill(~active, -torch.inf).amax(dim=(-1, -2))
            pixels = logits[index][valid[index]]
            statistics = torch.stack((pixels.mean(), pixels.max()))[None]
            decisions.append(self.output(torch.cat((mean, maximum, statistics), dim=1)).reshape(()))
        return torch.stack(decisions)


class DecisionModel(nn.Module):
    def __init__(self, candidate="control", base_channels=16, head_seed=9_000_042):
        super().__init__()
        self.candidate = candidate
        self.model_kind, self.score_definition = identity(candidate)
        # Matches SegmentationOnlyModel's exact construction and parameter names.
        self.segmentation = SmallUNet(3, 1, base_channels)
        if candidate == "decision_head":
            # Head initialization must not consume backbone/training global RNG.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator().manual_seed(head_seed).get_state())
                self.head = ContextHead(base_channels * 8)

    def forward(self, image, valid):
        skips = []
        x = image
        for index, encoder in enumerate(self.segmentation.encoders):
            if index:
                x = F.avg_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)
        bottleneck = x
        for decoder, skip in zip(self.segmentation.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        logits = self.segmentation.output(x)
        scores = (self.head(bottleneck, logits, valid) if self.candidate == "decision_head"
                  else valid_anomaly_score(logits, valid))
        return logits, scores


def make_model(cfg):
    return DecisionModel(cfg["candidate"], cfg["base_channels"], cfg["head_seed"])


def parameter_counts(model):
    segmentation = sum(p.numel() for p in model.segmentation.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"segmentation": segmentation, "decision_head": total - segmentation, "total": total}


def check_score_identity(checkpoint, cfg):
    kind, score = identity(cfg["candidate"])
    expected = {"candidate": cfg["candidate"], "base_channels": cfg["base_channels"], "head_seed": cfg["head_seed"]}
    if (checkpoint.get("model_kind") != kind or checkpoint.get("score_definition") != score
            or checkpoint.get("decision_model_config") != expected):
        raise ValueError("Checkpoint model/score identity differs from declared candidate")
