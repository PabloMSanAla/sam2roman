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
