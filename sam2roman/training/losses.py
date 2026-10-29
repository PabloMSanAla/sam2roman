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


class MultiClassCombinedLoss(nn.Module):
    """
    Combined Cross-Entropy and Multi-Class Dice Loss for boundary-aware UNet++ training.

    Parameters
    ----------
    class_weights : Optional[torch.Tensor]
        Optional class weighting for CrossEntropy (e.g. [1.0, 2.5, 4.0] for bg, interior, edge).
    ce_weight : float, default 1.0
        Weight factor for CrossEntropy loss.
    dice_weight : float, default 1.0
        Weight factor for Dice loss.
    smooth : float, default 1.0
        Laplace smoothing term for Dice calculation.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer("class_weights", class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Logits of shape (B, num_classes, H, W).
        targets : torch.Tensor
            Ground truth class labels of shape (B, H, W) with dtype int64.
        """
        # CrossEntropy
        ce = self.ce_loss(pred_logits, targets)

        # Multi-class Dice
        probs = F.softmax(pred_logits, dim=1)
        num_classes = pred_logits.shape[1]

        # One-hot encode targets to (B, num_classes, H, W)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_per_class)

        return self.ce_weight * ce + self.dice_weight * dice_loss

