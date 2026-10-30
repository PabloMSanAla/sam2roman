"""
Astronomical Boundary Segmentation Validation Metrics.
=====================================================

Provides distance-tolerant and soft continuous metrics designed specifically for thin
astronomical source boundaries (e.g., 1-2 pixel contact interfaces and stellar boundaries):
1. Relaxed / Distance-Tolerant Boundary IoU:
   Allows a spatial tolerance window of tau pixels (default: 1 or 2 pixels).
   Prevents strict exact-pixel argmax penalties (where a 1-pixel shift yields 0.0 IoU).
2. Continuous Soft IoU (Generalized Soft Jaccard):
   Evaluates smooth overlap between continuous probability distributions without argmax thresholding.
3. Strict Binary IoU:
   Standard 0-slack Jaccard index retained for comparative benchmarking.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F


def dilate_binary_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """
    Dilate a 2D or 3D binary mask on GPU/CPU using fast 2D max-pooling.

    Parameters
    ----------
    mask : torch.Tensor
        Binary boolean or float tensor of shape (B, H, W) or (B, 1, H, W).
    radius : int, default 1
        Dilation radius in pixels (kernel size = 2 * radius + 1).

    Returns
    -------
    torch.Tensor
        Dilated boolean mask matching input dimensions.
    """
    if radius <= 0:
        return mask > 0

    ksize = 2 * radius + 1
    orig_ndim = mask.ndim
    if orig_ndim == 3:
        m = mask.unsqueeze(1).float()
    else:
        m = mask.float()

    dilated = F.max_pool2d(m, kernel_size=ksize, stride=1, padding=radius) > 0.5
    return dilated.squeeze(1) if orig_ndim == 3 else dilated


def compute_relaxed_boundary_iou(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    tolerance: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Distance-Tolerant Boundary IoU with spatial pixel slack tolerance.

    A predicted boundary pixel is accepted as a true positive if it lies within
    `tolerance` pixels of any ground truth boundary pixel (and vice versa).

    Parameters
    ----------
    pred_mask : torch.Tensor
        Boolean or binary tensor of shape (B, H, W) indicating predicted boundary pixels.
    target_mask : torch.Tensor
        Boolean or binary tensor of shape (B, H, W) indicating ground truth boundary pixels.
    tolerance : int, default 1
        Pixel distance tolerance window (1 or 2 pixels recommended for 2-pixel edges).
    eps : float, default 1e-6
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Per-sample relaxed IoU tensor of shape (B,).
    """
    pred_b = pred_mask > 0
    tgt_b = target_mask > 0

    if tolerance <= 0:
        inter = torch.sum(pred_b & tgt_b, dim=(-2, -1)).float()
        union = torch.sum(pred_b | tgt_b, dim=(-2, -1)).float()
        both_empty = (pred_b.sum(dim=(-2, -1)) == 0) & (tgt_b.sum(dim=(-2, -1)) == 0)
        iou = (inter + eps) / (union + eps)
        return torch.where(both_empty, torch.ones_like(iou), iou)

    # Dilate both masks by tolerance window
    tgt_dilated = dilate_binary_mask(tgt_b, radius=tolerance)
    pred_dilated = dilate_binary_mask(pred_b, radius=tolerance)

    # True positives: predictions within tolerance of target, and targets covered by prediction
    tp_pred = torch.sum(pred_b & tgt_dilated, dim=(-2, -1)).float()
    tp_tgt = torch.sum(tgt_b & pred_dilated, dim=(-2, -1)).float()

    inter_tol = 0.5 * (tp_pred + tp_tgt)
    p_sum = torch.sum(pred_b, dim=(-2, -1)).float()
    t_sum = torch.sum(tgt_b, dim=(-2, -1)).float()
    union_tol = p_sum + t_sum - inter_tol

    both_empty = (p_sum == 0) & (t_sum == 0)
    iou = (inter_tol + eps) / (union_tol + eps)
    return torch.where(both_empty, torch.ones_like(iou), iou)


def compute_soft_continuous_iou(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Continuous Soft IoU (Jaccard derived from generalized Soft Dice).

    Evaluates continuous probability fields in [0, 1] without hard argmax thresholding.
    J = D / (2 - D) where D = 2 * sum(P * T) / (sum(P^2) + sum(T^2)).

    Parameters
    ----------
    pred_probs : torch.Tensor
        Continuous prediction probabilities of shape (B, H, W) or (B, C, H, W) in [0, 1].
    target_probs : torch.Tensor
        Continuous target probabilities of shape (B, H, W) or (B, C, H, W) in [0, 1].
    eps : float, default 1e-6
        Numerical stability epsilon.

    Returns
    -------
    torch.Tensor
        Continuous Soft IoU tensor matching batch/class dimensions.
    """
    inter = torch.sum(pred_probs * target_probs, dim=(-2, -1))
    p_sq = torch.sum(pred_probs ** 2, dim=(-2, -1))
    t_sq = torch.sum(target_probs ** 2, dim=(-2, -1))

    dice = (2.0 * inter + eps) / (p_sq + t_sq + eps)
    soft_iou = dice / torch.clamp(2.0 - dice, min=eps)
    return soft_iou


def evaluate_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    boundary_tolerance: int = 1,
) -> Dict[str, float]:
    """
    Compute comprehensive validation metrics for a mini-batch.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (B, num_classes, H, W).
    targets : torch.Tensor
        Ground truth targets: either discrete (B, H, W) or continuous soft targets (B, 3, H, W).
    boundary_tolerance : int, default 1
        Pixel distance tolerance for relaxed boundary IoU (tau=1 or 2 px).

    Returns
    -------
    Dict[str, float]:
        - 'interior_iou': Core/interior IoU (class 1)
        - 'relaxed_boundary_iou': Distance-tolerant boundary IoU with tau pixels slack (class 2)
        - 'strict_boundary_iou': Strict 0-slack boundary IoU (class 2)
        - 'soft_boundary_iou': Continuous Soft IoU of boundary probability channel
        - 'macro_iou': Mean of interior_iou and relaxed_boundary_iou
    """
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    num_classes = logits.shape[1]

    # Handle soft vs discrete targets
    if targets.ndim == 4:
        target_classes = torch.argmax(targets, dim=1)
        targets_soft = targets.float()
    else:
        target_classes = targets
        targets_soft = (
            F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

    # 1. Interior IoU (Class 1)
    p_int = (preds == 1)
    t_int = (target_classes == 1)
    int_inter = torch.sum(p_int & t_int).item()
    int_union = torch.sum(p_int | t_int).item()
    interior_iou = (int_inter + 1e-6) / (int_union + 1e-6)

    # 2. Strict Boundary IoU (Class 2, 0 slack)
    p_bound = (preds == 2)
    t_bound = (target_classes == 2)
    bound_inter = torch.sum(p_bound & t_bound).item()
    bound_union = torch.sum(p_bound | t_bound).item()
    strict_boundary_iou = (bound_inter + 1e-6) / (bound_union + 1e-6)

    # 3. Relaxed / Distance-Tolerant Boundary IoU (Class 2, tau slack)
    relaxed_iou_tensor = compute_relaxed_boundary_iou(
        p_bound, t_bound, tolerance=boundary_tolerance
    )
    relaxed_boundary_iou = relaxed_iou_tensor.mean().item()

    # 4. Continuous Soft IoU (Class 2)
    soft_iou_tensor = compute_soft_continuous_iou(
        probs[:, 2, :, :], targets_soft[:, 2, :, :]
    )
    soft_boundary_iou = soft_iou_tensor.mean().item()

    # 5. Macro IoU (Interior + Relaxed Boundary)
    macro_iou = 0.5 * (interior_iou + relaxed_boundary_iou)

    return {
        "interior_iou": interior_iou,
        "relaxed_boundary_iou": relaxed_boundary_iou,
        "strict_boundary_iou": strict_boundary_iou,
        "soft_boundary_iou": soft_boundary_iou,
        "macro_iou": macro_iou,
    }
