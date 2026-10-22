"""
Segmentation loss functions for astronomical source extraction and fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation, robust to class imbalance (e.g. small astronomical sources).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_logits : torch.Tensor
            Unnormalized logits from mask decoder, shape (N, ...).
        targets : torch.Tensor
            Binary ground truth masks (0 or 1), shape (N, ...).
        """
        probs = torch.sigmoid(pred_logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()

        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss to down-weight easy background examples and focus on faint astronomical sources.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets.float(), reduction="none")
        probs = torch.sigmoid(pred_logits)
        p_t = probs * targets.float() + (1 - probs) * (1 - targets.float())
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        return (focal_weight * bce).mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined BCE with Logits and Dice Loss for robust SAM 2 mask decoder fine-tuning.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(pred_logits, targets.float())
        dice = self.dice_loss(pred_logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
