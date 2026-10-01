"""Losses for clean reconstruction and known synthetic defect masks."""

import torch
from torch.nn import functional as F


def inspection_loss(reconstruction, logits, clean, mask):
    reconstruction_loss = F.l1_loss(reconstruction, clean)
    bce = F.binary_cross_entropy_with_logits(
        logits, mask, pos_weight=logits.new_tensor(3.0)
    )
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * mask).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = reconstruction_loss + bce + dice
    return total, {
        "reconstruction": reconstruction_loss.detach(),
        "segmentation_bce": bce.detach(),
        "segmentation_dice": dice.detach(),
    }
