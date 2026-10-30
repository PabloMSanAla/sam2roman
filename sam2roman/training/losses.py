"""
Segmentation loss functions for astronomical source extraction and fine-tuning.
Includes class-imbalance robust losses: Balanced Boundary Composite Loss, Focal Loss, and Dice Loss.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
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


class BalancedBoundaryCompositeLoss(nn.Module):
    """
    Balanced Composite Loss for astronomical source and boundary segmentation.
    Designed specifically to overcome severe class imbalance (~97% sky background, ~0.5% edge)
    and optimization instability:

    1. Multi-Class Focal Loss (gamma=2.0) with edge prioritization (alpha=[0.5, 2.5, 8.0]):
       Exponentially suppresses confident background gradients by up to 10,000x while maintaining
       strong gradients on rare boundary and core pixels.
    2. Class-Weighted Soft Dice Loss:
       Allocates class weights (default [0.1, 0.4, 0.5]) so that boundary delineation receives 50%
       of the spatial overlap gradient, preventing background from dominating.
    3. Dual Support for Hard & Soft Targets:
       Seamlessly accepts discrete class labels (B, H, W) in {0, 1, 2} or continuous soft-boundary
       probability fields (B, 3, H, W).
    4. Numerical Stability Safeguards:
       Logit clamping [-30.0, 30.0], Laplace epsilon smoothing, and torch.nan_to_num guard.
    """

    def __init__(
        self,
        class_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        dice_class_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        logit_clamp: float = 30.0,
    ):
        super().__init__()
        if class_weights is None:
            # Heavily prioritize edge (class 2: 8.0) and interior (class 1: 2.5) over sky (class 0: 0.5)
            cw = [0.5, 2.5, 8.0]
        elif isinstance(class_weights, (list, tuple)):
            cw = list(class_weights)
        else:
            cw = class_weights.detach().cpu().tolist()

        if dice_class_weights is None:
            # 50% dice weight on edge, 40% on core, 10% on background
            dw = [0.1, 0.4, 0.5]
        elif isinstance(dice_class_weights, (list, tuple)):
            dw = list(dice_class_weights)
        else:
            dw = dice_class_weights.detach().cpu().tolist()

        self.register_buffer("class_weights", torch.tensor(cw, dtype=torch.float32))
        self.register_buffer("dice_class_weights", torch.tensor(dw, dtype=torch.float32))
        self.gamma = float(gamma)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.logit_clamp = float(logit_clamp)

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute balanced composite loss.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Logits of shape (B, num_classes, H, W).
        targets : torch.Tensor
            Ground truth targets: either discrete (B, H, W) int64 or continuous soft targets (B, num_classes, H, W).
        """
        # Mixed precision safeguard: clamp logits to prevent exp() overflow
        clamped_logits = torch.clamp(pred_logits, -self.logit_clamp, self.logit_clamp)
        log_probs = F.log_softmax(clamped_logits, dim=1)
        probs = torch.exp(log_probs)
        num_classes = clamped_logits.shape[1]

        # Convert discrete targets (B, H, W) to one-hot (B, num_classes, H, W)
        if targets.ndim == 3:
            targets_one_hot = (
                F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
        elif targets.ndim == 4:
            targets_one_hot = targets.float()
        else:
            raise ValueError(f"Unsupported target shape: {targets.shape}")

        # Ensure class weights match device and num_classes
        if self.class_weights.numel() != num_classes:
            cw = torch.ones(num_classes, device=clamped_logits.device, dtype=clamped_logits.dtype)
        else:
            cw = self.class_weights.to(clamped_logits.device, dtype=clamped_logits.dtype)

        if self.dice_class_weights.numel() != num_classes:
            dw = torch.ones(num_classes, device=clamped_logits.device, dtype=clamped_logits.dtype)
        else:
            dw = self.dice_class_weights.to(clamped_logits.device, dtype=clamped_logits.dtype)

        # 1. Multi-Class Edge-Prioritized Focal Loss
        # Down-weight confident pixels: (1 - p_t)^gamma
        focal_modulator = torch.pow(1.0 - probs, self.gamma)
        alpha_t = cw.view(1, num_classes, 1, 1)
        pixel_focal = - (alpha_t * focal_modulator * targets_one_hot * log_probs)
        focal_loss = pixel_focal.sum(dim=1).mean()

        # 2. Class-Weighted Soft Dice Loss
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        weighted_dice = (dw * dice_per_class).sum() / (dw.sum() + 1e-7)
        dice_loss = 1.0 - weighted_dice

        # 3. Composite combination with NaN guard
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return torch.nan_to_num(total_loss, nan=0.0, posinf=50.0, neginf=0.0)


class MultiClassCombinedLoss(nn.Module):
    """
    Combined Loss for boundary-aware UNet++ training.
    Defaults to BalancedBoundaryCompositeLoss with edge prioritization and soft target support.

    Parameters
    ----------
    class_weights : Optional[torch.Tensor]
        Class weighting for Focal/CE loss (e.g. [0.5, 2.5, 8.0] for bg, interior, edge).
    ce_weight : float, default 1.0
        Weight factor for Focal/CE loss.
    dice_weight : float, default 1.0
        Weight factor for Dice loss.
    smooth : float, default 1.0
        Laplace smoothing term for Dice calculation.
    gamma : float, default 2.0
        Focal modulating exponent.
    use_focal : bool, default True
        Whether to use Focal loss modulating factor to downweight easy background.
    """

    def __init__(
        self,
        class_weights: Optional[Union[Sequence[float], torch.Tensor]] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        gamma: float = 2.0,
        use_focal: bool = True,
    ):
        super().__init__()
        self.balanced_loss = BalancedBoundaryCompositeLoss(
            class_weights=class_weights if class_weights is not None else [0.5, 2.5, 8.0],
            dice_class_weights=[0.1, 0.4, 0.5],
            gamma=gamma if use_focal else 0.0,
            focal_weight=ce_weight,
            dice_weight=dice_weight,
            smooth=smooth,
        )

    def forward(
        self,
        pred_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.balanced_loss(pred_logits, targets)
