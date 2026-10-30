"""Unit tests for astronomical boundary evaluation metrics."""

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from sam2roman.training.metrics import (
    dilate_binary_mask,
    compute_relaxed_boundary_iou,
    compute_soft_continuous_iou,
    evaluate_batch_metrics,
)


def test_dilate_binary_mask():
    mask = torch.zeros(1, 16, 16, dtype=torch.bool)
    mask[0, 8, 8] = True

    # Radius 1 -> 3x3 block around (8, 8)
    dilated_1 = dilate_binary_mask(mask, radius=1)
    assert dilated_1[0, 8, 8]
    assert dilated_1[0, 7, 8]
    assert dilated_1[0, 9, 8]
    assert dilated_1[0, 8, 7]
    assert dilated_1[0, 8, 9]
    assert not dilated_1[0, 6, 8]  # 2 px away is False
    assert dilated_1.sum().item() == 9

    # Radius 0 -> identical mask
    dilated_0 = dilate_binary_mask(mask, radius=0)
    assert dilated_0.sum().item() == 1


def test_relaxed_boundary_iou_distance_tolerance():
    # Construct 64x64 ring boundary
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    dist = torch.sqrt((y - 32.0) ** 2 + (x - 32.0) ** 2)
    target = ((dist >= 14) & (dist <= 16)).unsqueeze(0)  # 2px ring

    # 1. Exact match
    pred_exact = target.clone()
    iou_exact = compute_relaxed_boundary_iou(pred_exact, target, tolerance=1)
    assert pytest.approx(iou_exact.item(), abs=1e-4) == 1.0

    # 2. 1-pixel shift (dist 15 to 17)
    pred_shift1 = ((dist >= 15) & (dist <= 17)).unsqueeze(0)
    strict_iou = compute_relaxed_boundary_iou(pred_shift1, target, tolerance=0)
    relaxed_iou_1px = compute_relaxed_boundary_iou(pred_shift1, target, tolerance=1)

    # Strict IoU drops dramatically for thin rings
    assert strict_iou.item() < 0.5
    # Relaxed IoU with tau=1 allows 1-pixel shift and scores 1.0
    assert pytest.approx(relaxed_iou_1px.item(), abs=1e-4) == 1.0

    # 3. 2-pixel shift (dist 16 to 18)
    pred_shift2 = ((dist >= 16) & (dist <= 18)).unsqueeze(0)
    strict_iou_2px = compute_relaxed_boundary_iou(pred_shift2, target, tolerance=0)
    relaxed_iou_2px = compute_relaxed_boundary_iou(pred_shift2, target, tolerance=2)
    assert strict_iou_2px.item() < 0.05
    assert pytest.approx(relaxed_iou_2px.item(), abs=1e-4) == 1.0

    # 4. Completely disjoint prediction
    pred_disjoint = ((dist >= 4) & (dist <= 6)).unsqueeze(0)
    iou_disjoint = compute_relaxed_boundary_iou(pred_disjoint, target, tolerance=1)
    assert iou_disjoint.item() < 0.01


def test_soft_continuous_iou():
    # Probability distribution
    p = torch.zeros(2, 32, 32)
    p[0, 10:20, 10:20] = 0.9
    p[1, 5:15, 5:15] = 0.8
    t = p.clone()

    iou_perfect = compute_soft_continuous_iou(p, t)
    assert torch.allclose(iou_perfect, torch.ones_like(iou_perfect), atol=1e-4)

    # Zero overlap
    t_zero = torch.zeros_like(p)
    t_zero[0, 0:5, 0:5] = 0.9
    iou_zero = compute_soft_continuous_iou(p, t_zero)
    assert torch.all(iou_zero < 0.01)


def test_evaluate_batch_metrics():
    # (B=2, C=3, H=32, W=32)
    logits = torch.randn(2, 3, 32, 32)
    targets_discrete = torch.randint(0, 3, (2, 32, 32))

    metrics_discrete = evaluate_batch_metrics(logits, targets_discrete, boundary_tolerance=1)
    assert "interior_iou" in metrics_discrete
    assert "relaxed_boundary_iou" in metrics_discrete
    assert "strict_boundary_iou" in metrics_discrete
    assert "soft_boundary_iou" in metrics_discrete
    assert "macro_iou" in metrics_discrete

    # Relaxed IoU should always be >= strict IoU
    assert metrics_discrete["relaxed_boundary_iou"] >= metrics_discrete["strict_boundary_iou"] - 1e-6

    # Test with 4D continuous soft targets
    targets_soft = F.softmax(torch.randn(2, 3, 32, 32), dim=1)
    metrics_soft = evaluate_batch_metrics(logits, targets_soft, boundary_tolerance=1)
    assert 0.0 <= metrics_soft["soft_boundary_iou"] <= 1.0
