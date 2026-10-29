"""
Unit tests for WebDataset offline materialization, streaming loader,
and GPU-accelerated astronomical transforms.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from astropy.io import fits

from sam2roman.tools.build_webdataset import build_webdataset
from sam2roman.training.webdataset_loader import create_webdataset_loader, resolve_shard_urls
from sam2roman.training.gpu_transforms import GPULeakySB, GPUAstroAugmentation, GPUAstroTransform
from sam2roman.visualization.stretch import leakySB
from sam2roman.training.unetpp_trainer import UNetPPTrainer


@pytest.fixture
def synthetic_fits_dir(tmp_path):
    """Generate minimal synthetic FITS images and masks for sharding test."""
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create 2 synthetic exposure pairs of size 1024x1024
    for i in range(2):
        img_arr = np.random.normal(loc=100.0, scale=5.0, size=(1024, 1024)).astype(np.float32)
        mask_arr = np.zeros((1024, 1024), dtype=np.int32)

        # Add 3 bright sources
        sources = [(256, 256, 15), (700, 300, 20), (512, 800, 25)]
        for lbl, (cy, cx, rad) in enumerate(sources, start=1):
            y, x = np.ogrid[:1024, :1024]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            src_mask = dist <= rad
            mask_arr[src_mask] = lbl
            img_arr[src_mask] += 1000.0 * np.exp(-0.5 * (dist[src_mask] / (rad / 2)) ** 2)

        # Save to FITS
        fits.writeto(images_dir / f"exp_{i:02d}.fits", img_arr)
        fits.writeto(masks_dir / f"exp_{i:02d}.fits", mask_arr)

    return images_dir, masks_dir


def test_build_webdataset_and_loader(synthetic_fits_dir, tmp_path):
    images_dir, masks_dir = synthetic_fits_dir
    output_dir = tmp_path / "webdataset_shards"

    # 1. Build shards with chunk_size=512
    summary = build_webdataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        chunk_size=512,
        stride=256,
        min_objects=1,
        val_ratio=0.5,
        max_samples_per_shard=10,
    )

    assert summary["train_cutouts"] > 0
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "train").exists()

    # 2. Test streaming loader
    train_shards = output_dir / "train"
    loader = create_webdataset_loader(
        urls=train_shards,
        batch_size=2,
        is_train=True,
        shuffle_buffer=10,
        num_workers=0,
    )

    batch = next(iter(loader))
    assert "raw" in batch
    assert "target" in batch
    assert "meta" in batch

    # Verify tensor shapes & types
    assert batch["raw"].shape == (2, 1, 512, 512)
    assert batch["raw"].dtype == torch.float32
    assert batch["target"].shape == (2, 512, 512)
    assert batch["target"].dtype == torch.int64

    # Verify target labels are in {0, 1, 2}
    unique_labels = torch.unique(batch["target"]).tolist()
    for lbl in unique_labels:
        assert lbl in (0, 1, 2)


def test_gpu_leaky_sb_parity():
    """Verify GPULeakySB matches CPU leakySB analytical formulation."""
    np.random.seed(42)
    raw_np = np.random.uniform(-10.0, 5000.0, (256, 256)).astype(np.float32)

    # CPU reference with preserve_color_ratios=False
    cpu_ref = leakySB(
        raw_np, raw_np, raw_np,
        zero_points=31.4, pixel_scale=0.11, mmin=13.0, mmax=32.0,
        preserve_color_ratios=False,
    )

    # GPULeakySB module
    gpu_module = GPULeakySB(zero_points=31.4, pixel_scale=0.11, mmin=13.0, mmax=32.0, out_channels=3)
    raw_t = torch.from_numpy(raw_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
    gpu_out = gpu_module(raw_t).permute(0, 2, 3, 1).squeeze(0).numpy()

    # Values must match within float32 precision
    max_diff = np.max(np.abs(cpu_ref - gpu_out))
    assert max_diff < 1e-5, f"Difference between CPU and GPU stretch too high: {max_diff}"
    assert np.all(gpu_out >= 0.0) and np.all(gpu_out <= 1.0)


def test_gpu_astro_augmentation():
    """Verify in-VRAM noise injection and spatial augmentations."""
    aug = GPUAstroAugmentation(
        enable_noise=True,
        enable_spatial=True,
        shot_noise_prob=1.0,  # Force shot noise
        read_noise_prob=1.0,  # Force read noise
        sky_pedestal_prob=1.0, # Force pedestal shift
        flux_scale_prob=1.0,  # Force flux scaling
        rot_prob=1.0,
        flip_h_prob=1.0,
        flip_v_prob=1.0,
    )
    aug.train()

    raw = torch.full((4, 1, 64, 64), 500.0, dtype=torch.float32)
    target = torch.ones((4, 64, 64), dtype=torch.int64)

    aug_raw, aug_target = aug(raw, target)

    assert aug_raw.shape == raw.shape
    assert aug_target.shape == target.shape
    # Raw flux should be modified by noise and scaling
    assert not torch.allclose(aug_raw, raw)

    # In eval mode, should be passthrough
    aug.eval()
    eval_raw, eval_target = aug(raw, target)
    assert torch.allclose(eval_raw, raw)
    assert torch.allclose(eval_target, target)


def test_unetpp_webdataset_end_to_end(synthetic_fits_dir, tmp_path):
    """Verify UNetPPTrainer trains end-to-end on WebDataset shards with GPU transforms."""
    images_dir, masks_dir = synthetic_fits_dir
    shards_dir = tmp_path / "train_shards"

    # Build 512x512 shards
    build_webdataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=shards_dir,
        chunk_size=512,
        stride=256,
        min_objects=1,
        val_ratio=0.5,
    )

    trainer = UNetPPTrainer(
        in_channels=3,
        num_classes=3,
        features=(8, 16, 32, 64, 128),
        learning_rate=1e-3,
        gpu_transform=True,
    )

    history = trainer.train(
        dataset=shards_dir / "train",
        val_dataset=shards_dir / "val",
        epochs=2,
        batch_size=2,
        num_workers=0,
        verbose=False,
    )

    assert len(history["train_loss"]) == 2
    assert history["train_loss"][0] > 0
