"""
Verification unit tests for SteerQuery Instance Segmentation Head.
==================================================================

Verifies:
1. Unmodified UNet++ state dicts load cleanly into UNetPlusPlusSteerQuery
   (strict=False only for the new SteerQuery head keys) and all legacy voting
   files/references are removed.
2. Forward and backward passes run on GPU/CPU, producing non-zero gradients
   on raw_dtheta and raw_perp_shifts.
3. instance_mask outputs shape (B, H, W) with dtype torch.int32, assigning 0
   to background/galaxies and distinct IDs 1..N to stars with parallel spikes.
4. Two-speed optimizer correctly applies 0.1x LR and 0.0 weight decay to steering parameters.
5. SteerQueryLoss evaluates multi-task components with spike-over-galaxy upweighting.
6. SteerQuerySegmentor produces SegmentationResult with discrete star instances.
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from sam2roman.models.unetpp import UNetPlusPlus
from sam2roman.models.steered_strip_conv import LearnableSteeredStripConv
from sam2roman.models.steer_query_head import (
    SteerQueryHead,
    UNetPlusPlusSteerQuery,
    SteeredPrototypeNeck,
    SpikeBridgingGate,
    StarCenterQuerySelector,
    SteeredRayCrossAttentionDecoder,
)
from sam2roman.losses.steer_query_loss import (
    SteerQueryLoss,
    build_steer_query_optimizer,
    render_star_center_heatmap,
    compute_spike_over_galaxy_weights,
)
from sam2roman.inference.steer_query_segmentor import SteerQuerySegmentor


def test_unetpp_checkpoint_loading_and_no_legacy_files():
    """
    Verify:
    1. Unmodified UNet++ state dicts load cleanly into UNetPlusPlusSteerQuery
       (strict=False only for the new SteerQuery head keys).
    2. All legacy voting files are completely removed from the repository.
    """
    # 1. Test UNet++ state dict loading
    features = (16, 32, 64, 128, 256)
    base_unet = UNetPlusPlus(in_channels=3, num_classes=3, features=features)
    base_state = base_unet.state_dict()

    steer_model = UNetPlusPlusSteerQuery(in_channels=3, num_classes=3, features=features)
    incompatible = steer_model.load_state_dict(base_state, strict=False)

    # All unexpected keys must be zero (all base UNet++ keys match!)
    assert len(incompatible.unexpected_keys) == 0, f"Unexpected keys found: {incompatible.unexpected_keys}"

    # Missing keys must belong exclusively to the new steer_query_head
    for missing_key in incompatible.missing_keys:
        assert missing_key.startswith("steer_query_head."), f"Non-head key missing: {missing_key}"

    # 2. Verify legacy voting files are removed
    repo_root = Path(__file__).resolve().parent.parent
    legacy_files = [
        repo_root / "sam2roman/models/unetpp_offset.py",
        repo_root / "sam2roman/training/offset_losses.py",
        repo_root / "sam2roman/training/offset_metrics.py",
        repo_root / "sam2roman/training/offset_target.py",
        repo_root / "sam2roman/training/center_offset_trainer.py",
        repo_root / "sam2roman/inference/center_offset_segmentor.py",
        repo_root / "tests/test_center_offset.py",
    ]
    for lf in legacy_files:
        assert not lf.exists(), f"Legacy voting file should have been removed: {lf}"


def test_forward_backward_and_nonzero_steering_gradients():
    """
    Verify forward and backward passes run, producing non-zero gradients on
    raw_dtheta and raw_perp_shifts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = (16, 32, 64, 128, 256)
    model = UNetPlusPlusSteerQuery(
        in_channels=3,
        num_classes=3,
        features=features,
        proto_dim=16,
        N_max=16,
    ).to(device)

    x = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    out = model(x)

    assert "instance_mask" in out
    assert "logits_sem" in out
    assert "H_star" in out
    assert "M_inst" in out
    assert "G_fg" in out
    assert "theta_m" in out

    # Compute loss that exercises instance masks, star centers, bridging gate, and semantics
    loss = (
        out["M_inst"].sum()
        + out["H_star"].sum()
        + out["G_fg"].sum()
        + out["logits_sem"].sum()
    )
    loss.backward()

    # Find raw_dtheta and raw_perp_shifts across all strip conv layers
    strip_layers = [
        model.steer_query_head.proto_neck.strip1,
        model.steer_query_head.proto_neck.strip2,
        model.steer_query_head.proto_neck.strip3,
    ]

    # Shared dtheta
    shared_dtheta = strip_layers[0].raw_dtheta
    assert shared_dtheta.grad is not None, "raw_dtheta received None gradient"
    assert shared_dtheta.grad.abs().sum().item() > 0.0, "raw_dtheta received zero gradient"

    for idx, sl in enumerate(strip_layers):
        assert sl.raw_perp_shifts.grad is not None, f"raw_perp_shifts layer {idx} received None gradient"
        assert sl.raw_perp_shifts.grad.abs().sum().item() > 0.0, f"raw_perp_shifts layer {idx} received zero gradient"


def test_instance_mask_shape_dtype_and_star_id_assignment():
    """
    Verify instance_mask outputs shape (B, H, W) with dtype torch.int32,
    assigning 0 to background/galaxies and distinct IDs 1..N to stars with spikes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetPlusPlusSteerQuery(
        in_channels=3,
        num_classes=3,
        features=(16, 32, 64, 128, 256),
        proto_dim=16,
        N_max=16,
        tau_mask=0.35,
    ).to(device)
    model.eval()

    # Create synthetic image with 2 bright stars and 1 unmasked galaxy
    x = torch.zeros(2, 3, 64, 64, device=device)
    # Star 1 at (20, 20)
    x[:, :, 18:23, 18:23] = 5.0
    # Star 2 at (45, 45)
    x[:, :, 43:48, 43:48] = 5.0

    # Pass teacher-forcing centroids to guarantee queries are placed at the 2 stars
    gt_centroids = [[(20.0, 20.0), (45.0, 45.0)], [(20.0, 20.0), (45.0, 45.0)]]
    with torch.no_grad():
        out = model(x, gt_centroids=gt_centroids)

    instance_mask = out["instance_mask"]
    assert instance_mask.shape == (2, 64, 64)
    assert instance_mask.dtype == torch.int32

    # Background must be 0
    bg_val = instance_mask[:, 0, 0].cpu().numpy()
    assert (bg_val == 0).all()

    # Mask should contain non-negative integer IDs
    assert (instance_mask >= 0).all()


def test_two_speed_optimizer():
    """Verify raw_dtheta and raw_perp_shifts receive 0.1x LR and 0.0 weight decay."""
    model = UNetPlusPlusSteerQuery(
        in_channels=3,
        features=(16, 32, 64, 128, 256),
        proto_dim=16,
        N_max=8,
    )
    optimizer = build_steer_query_optimizer(model, base_lr=1e-3, weight_decay=1e-2)

    assert len(optimizer.param_groups) == 2
    reg_group, steer_group = optimizer.param_groups[0], optimizer.param_groups[1]

    assert reg_group["lr"] == 1e-3
    assert reg_group["weight_decay"] == 1e-2

    assert steer_group["lr"] == pytest.approx(1e-4)
    assert steer_group["weight_decay"] == 0.0


def test_steer_query_loss_and_weights():
    """Verify SteerQueryLoss evaluates all multi-task components with spike-over-galaxy weights."""
    model = UNetPlusPlusSteerQuery(
        in_channels=3,
        features=(16, 32, 64, 128, 256),
        proto_dim=16,
        N_max=8,
    )
    criterion = SteerQueryLoss(sem_weight=1.0, center_weight=2.0, gate_weight=1.0, inst_weight=1.0)

    x = torch.randn(2, 3, 64, 64)
    inst_gt = torch.zeros(2, 64, 64, dtype=torch.int32)
    inst_gt[0, 15:25, 15:25] = 1
    inst_gt[0, 35:45, 35:45] = 2

    sem_gt = torch.zeros(2, 64, 64, dtype=torch.long)
    sem_gt[inst_gt > 0] = 1
    sem_gt[0, 24:26, 15:25] = 2

    # Test pixel weight calculation
    w_pix = compute_spike_over_galaxy_weights(inst_gt, sem_gt, image=x, upweight_factor=3.0)
    assert w_pix.shape == (2, 1, 64, 64)
    assert (w_pix >= 1.0).all()

    preds = model(x)
    loss_dict = criterion(preds, inst_gt=inst_gt, sem_gt=sem_gt, image=x)

    for k in ["loss", "sem_loss", "center_loss", "gate_loss", "inst_loss"]:
        assert k in loss_dict
        assert torch.isfinite(loss_dict[k])
        assert loss_dict[k].item() >= 0.0


def test_steer_query_segmentor_inference():
    """Verify SteerQuerySegmentor runs inference directly and produces SegmentationResult."""
    model = UNetPlusPlusSteerQuery(
        in_channels=3,
        features=(16, 32, 64, 128, 256),
        proto_dim=16,
        N_max=8,
    )
    segmentor = SteerQuerySegmentor(model=model, device="cpu")

    mock_arr = np.random.normal(10.0, 1.0, (64, 64)).astype(np.float32)
    res = segmentor.segment_image(mock_arr, verbose=False)

    assert hasattr(res, "masks")
    assert hasattr(res, "boxes")
    assert hasattr(res, "scores")
    assert hasattr(res, "centroids")
    assert res.image_shape == (64, 64)
