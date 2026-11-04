"""
Loss functions for Center Offset Voting instance segmentation.
==============================================================

Includes:
1. CenterFocalLoss: Penalty-reduced Gaussian focal loss for center heatmap keypoints (CenterNet).
2. MaskedSmoothL1Loss: Dense 2D offset vector regression restricted strictly to foreground pixels.
3. AngularCosineDistanceLoss: Directional cosine loss forcing offset vectors along long diffraction spikes
   to maintain exact ray alignment with the central stellar core.
4. CenterOffsetCompositeLoss: Multi-task weighted composite loss.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss


class CenterFocalLoss(nn.Module):
    """
    Penalty-reduced Gaussian focal loss for object center heatmap estimation.
    Paper: Zhou et al., "Objects as Points" (CenterNet, 2019).

    Parameters
    ----------
    alpha : float, default 2.0
        Focusing parameter suppressing easy non-peak examples.
    beta : float, default 4.0
        Gaussian penalty reduction parameter softening loss around peak neighbors.
    eps : float, default 1e-6
        Numerical epsilon clamping log probabilities.
    logit_clamp : float, default 15.0
        Maximum absolute logit threshold to prevent float underflow/overflow.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        eps: float = 1e-6,
        logit_clamp: float = 15.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.logit_clamp = float(logit_clamp)

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss on Gaussian center heatmaps in float32.

        Parameters
        ----------
        pred_logits : torch.Tensor
            Predicted center heatmap logits of shape (B, 1, H, W).
        targets : torch.Tensor
            Target Gaussian heatmap in [0.0, 1.0] of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor:
            Scalar loss.
        """
        # Always compute focal log-likelihoods in float32 to prevent bfloat16 underflow
        logits = pred_logits.float()
        targets = targets.float().clamp(0.0, 1.0)

        # Clamping logits prevents exp() / sigmoid() saturation to exact 1.0 or 0.0
        logits = torch.clamp(logits, -self.logit_clamp, self.logit_clamp)
        pred = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)

        pos_mask = targets.eq(1.0)
        neg_mask = targets.lt(1.0)

        pos_loss = torch.pow(1.0 - pred, self.alpha) * torch.log(pred)
        neg_loss = (
            torch.pow(1.0 - targets, self.beta)
            * torch.pow(pred, self.alpha)
            * torch.log(1.0 - pred)
        )

        num_pos = pos_mask.sum().float()
        pos_term = (pos_loss * pos_mask.float()).sum()
        neg_term = (neg_loss * neg_mask.float()).sum()

        if num_pos > 0:
            loss = -(pos_term + neg_term) / num_pos
        else:
            # If no center peaks in the batch, normalize by batch size
            loss = -neg_term / max(1.0, float(targets.shape[0]))

        return torch.nan_to_num(loss, nan=0.0, posinf=50.0, neginf=0.0)


class MaskedSmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss computed strictly on foreground source pixels.

    Parameters
    ----------
    beta : float, default 1.0
        Threshold at which Smooth L1 switches between L1 and L2 penalty.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_offsets : torch.Tensor
            Predicted offsets (B, 2, H, W).
        target_offsets : torch.Tensor
            Target offsets (B, 2, H, W).
        fg_mask : torch.Tensor
            Binary foreground mask (B, 1, H, W) or (B, H, W) in {0, 1}.

        Returns
        -------
        torch.Tensor:
            Scalar offset regression loss.
        """
        pred_offsets = pred_offsets.float()
        target_offsets = target_offsets.float()
        fg_mask = fg_mask.float()
        if fg_mask.ndim == 3:
            fg_mask = fg_mask.unsqueeze(1)

        diff = F.smooth_l1_loss(pred_offsets, target_offsets, reduction="none", beta=self.beta)
        fg_expanded = fg_mask.expand_as(diff)

        num_fg_elements = fg_expanded.sum()
        if num_fg_elements > 0:
            loss = (diff * fg_expanded).sum() / num_fg_elements
        else:
            loss = pred_offsets.new_tensor(0.0)

        return torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)


class AngularCosineDistanceLoss(nn.Module):
    """
    Angular Cosine Distance Loss for offset vectors.
    Penalizes directional angular deviation along long astronomical diffraction spikes:
    loss = 1 - cos(theta) where cos(theta) = (pred . target) / (||pred|| * ||target|| + eps).

    Parameters
    ----------
    eps : float, default 1e-6
        Safeguard for zero magnitude vectors.
    min_norm : float, default 1.0
        Minimum pixel offset distance to apply angular constraint (avoids noisy core angles).
    """

    def __init__(self, eps: float = 1e-6, min_norm: float = 1.0):
        super().__init__()
        self.eps = float(eps)
        self.min_norm = float(min_norm)

    def forward(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_offsets : torch.Tensor
            Predicted offsets (B, 2, H, W).
        target_offsets : torch.Tensor
            Target offsets (B, 2, H, W).
        fg_mask : torch.Tensor
            Foreground mask (B, 1, H, W) or (B, H, W).

        Returns
        -------
        torch.Tensor:
            Scalar angular loss.
        """
        pred_offsets = pred_offsets.float()
        target_offsets = target_offsets.float()
        fg_mask = fg_mask.float()
        if fg_mask.ndim == 3:
            fg_mask = fg_mask.unsqueeze(1)

        pred_dy = pred_offsets[:, 0:1]
        pred_dx = pred_offsets[:, 1:2]
        tgt_dy = target_offsets[:, 0:1]
        tgt_dx = target_offsets[:, 1:2]

        dot = pred_dy * tgt_dy + pred_dx * tgt_dx
        norm_pred = torch.sqrt(pred_dy ** 2 + pred_dx ** 2 + self.eps)
        norm_tgt = torch.sqrt(tgt_dy ** 2 + tgt_dx ** 2 + self.eps)

        cos_sim = dot / (norm_pred * norm_tgt + self.eps)
        ang_dist = 1.0 - torch.clamp(cos_sim, -1.0, 1.0)

        # Apply constraint only on foreground pixels beyond minimum distance from center
        valid_mask = (fg_mask > 0) & (norm_tgt >= self.min_norm)
        valid_elements = valid_mask.sum().float()

        if valid_elements > 0:
            loss = (ang_dist * valid_mask.float()).sum() / valid_elements
        else:
            loss = pred_offsets.new_tensor(0.0)

        return torch.nan_to_num(loss, nan=0.0, posinf=2.0, neginf=0.0)


class CenterOffsetCompositeLoss(nn.Module):
    """
    Unified Multi-Task Composite Loss for Center Offset Segmentation:
    L_total = lambda_sem * L_sem + lambda_center * L_center + lambda_offset * L_offset + lambda_ang * L_ang

    Parameters
    ----------
    semantic_weight : float, default 1.0
        Weight for semantic foreground segmentation (BCE + Dice).
    center_weight : float, default 2.0
        Weight for center heatmap Gaussian focal loss.
    offset_weight : float, default 1.0
        Weight for masked Smooth L1 offset regression.
    angle_weight : float, default 0.5
        Weight for angular cosine direction loss (for diffraction spike deblending).
    center_alpha : float, default 2.0
        Focal loss alpha for center heatmaps.
    center_beta : float, default 4.0
        Focal loss beta for center heatmaps.
    """

    def __init__(
        self,
        semantic_weight: float = 1.0,
        center_weight: float = 2.0,
        offset_weight: float = 1.0,
        angle_weight: float = 0.5,
        center_alpha: float = 2.0,
        center_beta: float = 4.0,
    ):
        super().__init__()
        self.semantic_weight = float(semantic_weight)
        self.center_weight = float(center_weight)
        self.offset_weight = float(offset_weight)
        self.angle_weight = float(angle_weight)

        self.sem_loss_fn = CombinedSegmentationLoss(bce_weight=1.0, dice_weight=1.0)
        self.center_loss_fn = CenterFocalLoss(alpha=center_alpha, beta=center_beta)
        self.offset_loss_fn = MaskedSmoothL1Loss(beta=1.0)
        self.angle_loss_fn = AngularCosineDistanceLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        fg_target: torch.Tensor,
        center_target: torch.Tensor,
        offset_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss components.

        Parameters
        ----------
        predictions : Dict[str, torch.Tensor]
            Dictionary containing 'semantic', 'center', and 'offset' prediction tensors.
        fg_target : torch.Tensor
            Binary foreground mask (B, 1, H, W) or (B, H, W).
        center_target : torch.Tensor
            Gaussian center heatmap (B, 1, H, W).
        offset_target : torch.Tensor
            Ground truth 2D offsets (B, 2, H, W).

        Returns
        -------
        Dict[str, torch.Tensor]:
            Dictionary containing 'loss' (scalar total) and individual terms:
            'sem_loss', 'center_loss', 'offset_loss', 'angle_loss'.
        """
        sem_pred = predictions["semantic"].float()
        center_pred = predictions["center"].float()
        offset_pred = predictions["offset"].float()
        fg_target = fg_target.float()
        center_target = center_target.float()
        offset_target = offset_target.float()

        if fg_target.ndim == 3:
            fg_target = fg_target.unsqueeze(1)
        if center_target.ndim == 3:
            center_target = center_target.unsqueeze(1)

        # 1. Semantic loss
        l_sem = self.sem_loss_fn(sem_pred, fg_target)

        # 2. Center heatmap loss
        l_center = self.center_loss_fn(center_pred, center_target)

        # 3. Offset regression loss
        l_offset = self.offset_loss_fn(offset_pred, offset_target, fg_target)

        # 4. Angular direction loss (spike alignment)
        if self.angle_weight > 0.0:
            l_ang = self.angle_loss_fn(offset_pred, offset_target, fg_target)
        else:
            l_ang = offset_pred.new_tensor(0.0)

        total_loss = (
            self.semantic_weight * l_sem
            + self.center_weight * l_center
            + self.offset_weight * l_offset
            + self.angle_weight * l_ang
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=0.0)

        return {
            "loss": total_loss,
            "sem_loss": l_sem,
            "center_loss": l_center,
            "offset_loss": l_offset,
            "angle_loss": l_ang,
        }
