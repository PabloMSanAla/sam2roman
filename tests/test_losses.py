"""Unit tests for segmentation loss functions."""

import torch
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss, FocalLoss


def test_dice_loss_perfect_prediction():
    loss_fn = DiceLoss()
    logits = torch.ones((1, 1, 32, 32)) * 100.0  # sigmoid ~ 1
    targets = torch.ones((1, 1, 32, 32))

    loss = loss_fn(logits, targets)
    assert loss.item() < 1e-3


def test_dice_loss_inverse_prediction():
    loss_fn = DiceLoss()
    logits = torch.ones((1, 1, 32, 32)) * -100.0  # sigmoid ~ 0
    targets = torch.ones((1, 1, 32, 32))

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.95


def test_combined_loss():
    loss_fn = CombinedSegmentationLoss()
    logits = torch.randn((2, 1, 16, 16), requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16))

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert loss.item() > 0.0


def test_focal_loss():
    loss_fn = FocalLoss()
    logits = torch.randn((2, 1, 16, 16))
    targets = torch.randint(0, 2, (2, 1, 16, 16))
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0


def test_balanced_boundary_composite_loss():
    from sam2roman.training.losses import BalancedBoundaryCompositeLoss

    loss_fn = BalancedBoundaryCompositeLoss(
        class_weights=[0.5, 2.5, 8.0],
        dice_class_weights=[0.1, 0.4, 0.5],
        gamma=2.0,
    )

    # 1. Discrete targets (B, H, W)
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets_discrete = torch.randint(0, 3, (2, 32, 32))
    loss_discrete = loss_fn(logits, targets_discrete)
    assert torch.isfinite(loss_discrete)
    assert loss_discrete.item() > 0.0
    loss_discrete.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))

    # 2. Continuous soft targets (B, 3, H, W)
    logits_soft = torch.randn(2, 3, 32, 32, requires_grad=True)
    # Simulate a probability simplex (sum across channels == 1.0)
    raw_p = torch.rand(2, 3, 32, 32)
    targets_soft = raw_p / raw_p.sum(dim=1, keepdim=True)
    loss_soft = loss_fn(logits_soft, targets_soft)
    assert torch.isfinite(loss_soft)
    assert loss_soft.item() > 0.0
    loss_soft.backward()
    assert logits_soft.grad is not None
    assert torch.all(torch.isfinite(logits_soft.grad))

    # 3. Extreme logit safeguard (no NaN on huge logits)
    extreme_logits = torch.tensor([[[[100.0]], [[-100.0]], [[0.0]]]])
    tgt = torch.tensor([[[2]]])
    extreme_loss = loss_fn(extreme_logits, tgt)
    assert torch.isfinite(extreme_loss)

