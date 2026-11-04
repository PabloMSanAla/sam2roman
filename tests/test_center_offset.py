"""
Unit tests for Center Offset Voting instance segmentation.
==========================================================
"""

import numpy as np
import pytest
import torch

from sam2roman.models.unetpp_offset import (
    CrissCrossAttention,
    RecurrentCrissCrossAttention,
    UNetPlusPlusCenterOffset,
)
from sam2roman.training.offset_target import (
    compute_instance_centroids,
    generate_center_offset_targets_numpy,
    generate_center_offset_batch,
)
from sam2roman.training.offset_losses import (
    CenterFocalLoss,
    MaskedSmoothL1Loss,
    AngularCosineDistanceLoss,
    CenterOffsetCompositeLoss,
)
from sam2roman.training.offset_metrics import (
    extract_center_peaks,
    evaluate_center_detection,
    compute_offset_epe,
)
from sam2roman.inference.center_offset_segmentor import CenterOffsetSegmentor


def test_target_generation_numpy():
    """Verify mathematical correctness of center Gaussian peaks and offset vectors."""
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=np.int32)

    # Place an object (instance 1) centered at (20, 30) of radius 5
    cy_true, cx_true = 20, 30
    for y in range(cy_true - 5, cy_true + 6):
        for x in range(cx_true - 5, cx_true + 6):
            if (y - cy_true) ** 2 + (x - cx_true) ** 2 <= 25:
                mask[y, x] = 1

    fg, c_map, o_map, info = generate_center_offset_targets_numpy(mask, sigma=2.0)

    assert fg.shape == (h, w)
    assert c_map.shape == (h, w)
    assert o_map.shape == (2, h, w)

    # Centroid peak must be 1.0
    assert np.isclose(c_map[cy_true, cx_true], 1.0, atol=1e-3)
    # Background must be 0
    assert c_map[0, 0] == 0.0

    # Test offset at pixel (cy_true + 3, cx_true - 2)
    # delta_y = cy - y = 20 - 23 = -3
    # delta_x = cx - x = 30 - 28 = +2
    py, px = cy_true + 3, cx_true - 2
    assert np.isclose(o_map[0, py, px], -3.0, atol=1e-3)
    assert np.isclose(o_map[1, py, px], 2.0, atol=1e-3)

    # Background offset must be 0
    assert o_map[0, 0, 0] == 0.0
    assert o_map[1, 0, 0] == 0.0


def test_target_generation_batch():
    """Verify batch target generation for DataLoader collation."""
    b, h, w = 2, 32, 32
    masks = np.zeros((b, h, w), dtype=np.int32)
    masks[0, 10:15, 10:15] = 1
    masks[1, 20:25, 20:25] = 2

    fg, ctr, off = generate_center_offset_batch(masks, sigma=2.0)
    assert fg.shape == (b, 1, h, w)
    assert ctr.shape == (b, 1, h, w)
    assert off.shape == (b, 2, h, w)


def test_criss_cross_attention():
    """Verify Criss-Cross Attention shapes and gradient flow."""
    x = torch.randn(2, 32, 16, 16, requires_grad=True)
    cca = CrissCrossAttention(in_channels=32)
    out = cca(x)
    assert out.shape == (2, 32, 16, 16)
    out.sum().backward()
    assert x.grad is not None

    rcca = RecurrentCrissCrossAttention(in_channels=32, recurrence=2)
    out_r = rcca(x)
    assert out_r.shape == (2, 32, 16, 16)


def test_unetpp_center_offset_model():
    """Verify UNet++ model multi-task forward pass and deep supervision."""
    x = torch.randn(2, 3, 64, 64)
    model = UNetPlusPlusCenterOffset(
        in_channels=3,
        semantic_classes=1,
        features=(16, 32, 64, 128, 256),
        use_cca=True,
    )
    preds = model(x)
    assert "semantic" in preds
    assert "center" in preds
    assert "offset" in preds

    assert preds["semantic"].shape == (2, 1, 64, 64)
    assert preds["center"].shape == (2, 1, 64, 64)
    assert preds["offset"].shape == (2, 2, 64, 64)


def test_center_offset_losses():
    """Verify multi-task loss calculation and backward step."""
    pred_sem = torch.randn(2, 1, 32, 32, requires_grad=True)
    pred_ctr = torch.randn(2, 1, 32, 32, requires_grad=True)
    pred_off = torch.randn(2, 2, 32, 32, requires_grad=True)

    preds = {"semantic": pred_sem, "center": pred_ctr, "offset": pred_off}

    target_fg = torch.zeros(2, 1, 32, 32)
    target_fg[:, :, 10:20, 10:20] = 1.0

    target_ctr = torch.zeros(2, 1, 32, 32)
    target_ctr[:, :, 15, 15] = 1.0

    target_off = torch.zeros(2, 2, 32, 32)
    target_off[:, 0, 10:20, 10:20] = 2.0
    target_off[:, 1, 10:20, 10:20] = -3.0

    criterion = CenterOffsetCompositeLoss(
        semantic_weight=1.0,
        center_weight=2.0,
        offset_weight=1.0,
        angle_weight=0.5,
    )
    loss_dict = criterion(preds, target_fg, target_ctr, target_off)

    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].item() > 0

    loss_dict["loss"].backward()
    assert pred_sem.grad is not None
    assert pred_ctr.grad is not None
    assert pred_off.grad is not None


def test_metrics_evaluation():
    """Verify peak extraction, center matching, and offset EPE."""
    # Create synthetic heatmap with two peaks
    hm = np.zeros((64, 64), dtype=np.float32)
    hm[15, 20] = 0.95
    hm[45, 50] = 0.88

    peaks = extract_center_peaks(hm, threshold=0.3, min_distance=3)
    assert len(peaks) == 2

    gt_centroids = [(15.0, 20.0), (45.0, 50.0)]
    eval_res = evaluate_center_detection(peaks, gt_centroids, tolerance=2.0)
    assert eval_res["precision"] == 1.0
    assert eval_res["recall"] == 1.0
    assert eval_res["f1"] == 1.0
    assert eval_res["mae"] == 0.0

    # Offset EPE test (3D single cutout)
    pred_o = np.ones((2, 64, 64), dtype=np.float32) * 5.0
    target_o = np.ones((2, 64, 64), dtype=np.float32) * 2.0
    fg = np.ones((64, 64), dtype=np.float32)
    epe = compute_offset_epe(pred_o, target_o, fg)
    # sqrt((5-2)^2 + (5-2)^2) = sqrt(9 + 9) = 3*sqrt(2) approx 4.2426
    assert np.isclose(epe, 3.0 * np.sqrt(2.0), atol=1e-3)

    # Offset EPE test (4D batch with B=16)
    pred_batch = np.ones((16, 2, 64, 64), dtype=np.float32) * 5.0
    target_batch = np.ones((16, 2, 64, 64), dtype=np.float32) * 2.0
    fg_batch = np.ones((16, 1, 64, 64), dtype=np.float32)
    epe_batch = compute_offset_epe(pred_batch, target_batch, fg_batch)
    assert np.isclose(epe_batch, 3.0 * np.sqrt(2.0), atol=1e-3)


def test_inference_segmentor_mock():
    """Verify full inference segmentor on mock data."""
    model = UNetPlusPlusCenterOffset(
        in_channels=3,
        semantic_classes=1,
        features=(16, 32, 64, 128, 256),
        use_cca=False,
    )
    segmentor = CenterOffsetSegmentor(model=model, device="cpu")

    mock_image = np.random.rand(64, 64).astype(np.float32)
    res = segmentor.segment_image(mock_image, verbose=False)

    assert hasattr(res, "masks")
    assert hasattr(res, "boxes")
    assert hasattr(res, "scores")
    assert hasattr(res, "centroids")
    assert res.image_shape == (64, 64)
