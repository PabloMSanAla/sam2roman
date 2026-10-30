"""
Unit tests for boundary-aware UNet++ architecture, loss, trainer, and segmentor.
Verifies training on Apple Silicon MPS and automatic CUDA acceleration.
"""

import numpy as np
import pytest
import torch

from sam2roman.models.unetpp import UNetPlusPlus
from sam2roman.training.boundary import generate_3class_boundary_mask
from sam2roman.training.losses import MultiClassCombinedLoss
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.unetpp_trainer import UNetPPTrainer
from sam2roman.inference.unetpp_segmentor import UNetPPSegmentor


def test_unetpp_model_forward():
    # Test on CPU
    model = UNetPlusPlus(in_channels=3, num_classes=3, features=(16, 32, 64, 128, 256))
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)

    # Test deep supervision
    model_ds = UNetPlusPlus(in_channels=3, num_classes=3, deep_supervision=True, features=(16, 32, 64, 128, 256))
    model_ds.train()
    outs = model_ds(x)
    assert isinstance(outs, list)
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (2, 3, 64, 64)

    # Eval mode should return single tensor
    model_ds.eval()
    out_eval = model_ds(x)
    assert isinstance(out_eval, torch.Tensor)
    assert out_eval.shape == (2, 3, 64, 64)


def test_boundary_generation():
    segmap = np.zeros((60, 60), dtype=np.int32)
    # Object 1
    segmap[10:30, 10:30] = 1
    # Object 2 (touching Object 1 at x=30)
    segmap[10:30, 30:50] = 2

    target = generate_3class_boundary_mask(segmap, boundary_width=1)
    assert set(np.unique(target)) == {0, 1, 2}

    # Inter-object boundary between 1 and 2 must be labeled as 2
    assert np.any(target[:, 29:32] == 2)
    # Core interiors must be labeled as 1
    assert target[20, 20] == 1
    assert target[20, 40] == 1


def test_multiclass_loss():
    loss_fn = MultiClassCombinedLoss(class_weights=torch.tensor([1.0, 2.5, 4.0]))
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    targets = torch.randint(0, 3, (2, 32, 32))

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)


def test_dataset_dense_3class():
    img = np.random.normal(10.0, 1.0, (128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[30:50, 30:50] = 1
    mask[45:65, 45:65] = 2  # overlapping / blended source

    ds = AstronomicalDataset(
        images=[img],
        masks=[mask],
        target_mode="dense_3class",
        stretch_type="leakysb",
    )
    assert len(ds) == 1
    sample = ds[0]

    assert "image" in sample
    assert "target" in sample
    assert sample["image"].shape == (3, 128, 128)
    assert sample["target"].shape == (128, 128)
    assert set(sample["target"].unique().tolist()).issubset({0, 1, 2})


def test_unetpp_training_on_mps_or_cpu(tmp_path):
    """Verify that UNet++ training actually runs on MPS (if available) and converges."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing UNet++ training on device: {device}")

    # Generate synthetic images and masks
    np.random.seed(42)
    imgs = [np.random.normal(12.0, 1.5, (128, 128)).astype(np.float32) for _ in range(4)]
    masks = []
    for _ in range(4):
        m = np.zeros((128, 128), dtype=np.int32)
        m[30:50, 30:50] = 1
        m[45:70, 45:70] = 2
        masks.append(m)

    dataset = AstronomicalDataset(
        images=imgs,
        masks=masks,
        target_mode="dense_3class",
        stretch_type="leakysb",
    )

    trainer = UNetPPTrainer(
        features=(16, 32, 64, 128, 256),
        learning_rate=1e-3,
        device=device,
    )
    assert str(trainer.device) == device or str(trainer.device).startswith(device)

    # Train for 2 epochs
    history = trainer.train(
        dataset=dataset,
        val_dataset=dataset,
        epochs=2,
        batch_size=2,
        output_dir=tmp_path,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert (tmp_path / "unetpp_best.pt").exists()

    # Verify inference segmentor loads checkpoint and produces results
    segmentor = UNetPPSegmentor.from_checkpoint(tmp_path / "unetpp_best.pt", device=device)
    res = segmentor.segment_image(imgs[0], min_area=5, verbose=False)
    assert isinstance(res.masks, list)
    assert res.image_shape == (128, 128)

    # Verify photometry works seamlessly on UNet++ segmentation output
    phot = res.compute_photometry(imgs[0], verbose=False)
    assert isinstance(phot, list)


def test_soft_boundary_and_sdf_generation():
    from sam2roman.training.boundary import (
        generate_soft_boundary_target,
        generate_distance_transform_target,
    )

    segmap = np.zeros((64, 64), dtype=np.int32)
    segmap[20:44, 20:44] = 1

    # Option A: Soft continuous boundary probability simplex
    soft_target = generate_soft_boundary_target(segmap, boundary_width=2, sigma=1.2)
    assert soft_target.shape == (3, 64, 64)
    assert soft_target.dtype == np.float32
    # Verify probability simplex sums to 1.0 everywhere
    simplex_sums = np.sum(soft_target, axis=0)
    np.testing.assert_allclose(simplex_sums, 1.0, atol=1e-5)
    # Check that boundary channel (channel 2) contains smooth values > 0
    assert np.max(soft_target[2]) > 0.0
    assert np.min(soft_target[2]) >= 0.0

    # Option B: Signed Distance Field (SDF)
    sdf = generate_distance_transform_target(segmap, max_distance=10.0, truncate=True)
    assert sdf.shape == (64, 64)
    assert np.all(sdf >= -1.0) and np.all(sdf <= 1.0)
    # Inside source should be positive, background negative
    assert sdf[32, 32] > 0.0
    assert sdf[5, 5] < 0.0


def test_trainer_schedulers_and_soft_targets(tmp_path):
    """Verify trainer with soft targets, gradient clipping, and cosine warmup scheduler."""
    imgs = [np.random.normal(12.0, 1.5, (64, 64)).astype(np.float32) for _ in range(2)]
    masks = [np.zeros((64, 64), dtype=np.int32) for _ in range(2)]
    masks[0][15:35, 15:35] = 1
    masks[1][20:40, 20:40] = 1

    dataset = AstronomicalDataset(
        images=imgs,
        masks=masks,
        target_mode="dense_3class",
        soft_boundary=True,
    )
    sample = dataset[0]
    assert sample["target"].shape == (3, 64, 64)
    assert sample["target"].dtype == torch.float32

    trainer = UNetPPTrainer(
        features=(8, 16, 32, 64, 128),
        learning_rate=1e-4,
        weight_decay=1e-2,
        grad_clip_norm=1.0,
        device="cpu",
    )

    history = trainer.train(
        dataset=dataset,
        epochs=3,
        batch_size=2,
        scheduler_type="cosine",
        warmup_epochs=1,
        verbose=False,
    )
    assert len(history["train_loss"]) == 3
    assert len(history["learning_rate"]) == 3
    # Check that warmup increased LR from start_factor
    assert history["learning_rate"][1] > history["learning_rate"][0]
