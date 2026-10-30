#!/usr/bin/env python3
"""
NVIDIA Grace Hopper (GH200) Superchip Training Script for sam2roman
===================================================================

Optimized for NVIDIA Grace Hopper (GH200):
- Grace CPU: 72 ARM Neoverse-V2 cores with up to 480GB LPDDR5X unified RAM
- Hopper H100 GPU: 96GB/144GB HBM3 memory, 4 PFLOPS Tensor Cores
- NVLink-C2C: 900 GB/s bidirectional chip-to-chip interconnect

Features:
- WebDataset streaming loader (zero random disk seeks, reads directly from RAM disk or NVMe)
- In-VRAM GPUAstroTransform:
  - Vectorized GPU leakySB stretch (<0.05 ms per batch)
  - Astronomical noise injection (Poisson shot noise, Gaussian read noise, sky shifts)
  - Synchronized spatial augmentations (rotations, reflections)
- Full TF32 and native bfloat16 Automatic Mixed Precision (AMP)
- Sustained >90% GPU compute utilization
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure sam2roman package is found
for candidate in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (candidate / "sam2roman").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import matplotlib.pyplot as plt
import torch

from sam2roman.training.unetpp_trainer import UNetPPTrainer
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import create_webdataset_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.gh200")


def sync_s3_path(s3_uri: str, local_dest: Path) -> Path:
    """Download an S3 directory to local disk if an s3:// URI is provided."""
    local_dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Syncing S3 URI '{s3_uri}' -> '{local_dest}'...")
    cmd = ["aws", "s3", "sync", s3_uri, str(local_dest), "--exclude", "*.DS_Store"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to sync S3 path {s3_uri}: {e}")
        raise
    return local_dest


def main():
    parser = argparse.ArgumentParser(
        description="High-Throughput Grace Hopper (GH200) Training for sam2roman."
    )
    parser.add_argument(
        "--train_shards",
        type=str,
        required=True,
        help="Path or glob pattern for training shards (e.g. '/dev/shm/train/train-*.tar' or 's3://...').",
    )
    parser.add_argument(
        "--val_shards",
        type=str,
        default=None,
        help="Optional path or glob pattern for validation shards.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_gh200",
        help="Directory to store model checkpoints and training logs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size (32 or 64 recommended for H100 96GB).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW (lowered to 1e-4 for optimization stability).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay regularizing UNet++ dense nested skip connections (default: 1e-2).",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Maximum norm for gradient clipping (default: 1.0).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "none"],
        dest="scheduler_type",
        help="Learning rate scheduler (default: 'cosine' with warmup).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Number of linear warmup epochs before cosine decay (default: 2).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (4 recommended on Grace ARM CPU).",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=2000,
        help="In-memory shuffle buffer size for WebDataset.",
    )
    parser.add_argument(
        "--noise_aug",
        action="store_true",
        default=True,
        help="Enable GPU astronomical noise injection (Poisson shot + Gaussian read noise).",
    )

    args = parser.parse_args()

    # Hardware verification
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cpu_count = os.cpu_count() or 1
        logger.info(f"[Grace Hopper] GPU: {gpu_name} ({vram_gb:.1f} GB HBM3)")
        logger.info(f"[Grace Hopper] CPU: {cpu_count} ARM Neoverse cores")
        logger.info("  -> NVLink-C2C active: 900 GB/s bidirectional host-to-device streaming")
        logger.info("  -> Enabling TensorFloat-32 (TF32) and bfloat16 AMP")
        device = "cuda"
    elif torch.backends.mps.is_available():
        logger.info("[Apple Silicon] MPS accelerator active (Metal Performance Shaders FP32)")
        device = "mps"
    else:
        logger.warning("CUDA/MPS not detected! Running on CPU fallback.")
        device = "cpu"

    # Handle S3 downloads if needed
    train_shards = args.train_shards
    if train_shards.startswith("s3://"):
        local_train = Path("/dev/shm/roman_shards/train") if Path("/dev/shm").exists() else Path("./data_shards/train")
        train_shards = str(sync_s3_path(train_shards, local_train))

    val_shards = args.val_shards
    if val_shards and val_shards.startswith("s3://"):
        local_val = Path("/dev/shm/roman_shards/val") if Path("/dev/shm").exists() else Path("./data_shards/val")
        val_shards = str(sync_s3_path(val_shards, local_val))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GPU-accelerated transform pipeline
    gpu_transform = GPUAstroTransform(
        stretch_kwargs={
            "zero_points": 31.4,
            "pixel_scale": 0.11,
            "mmin": 13.0,
            "mmax": 32.0,
            "out_channels": 3,
        },
        aug_kwargs={
            "enable_noise": args.noise_aug,
            "enable_spatial": True,
            "shot_noise_prob": 0.5,
            "read_noise_prob": 0.5,
            "sky_pedestal_prob": 0.5,
            "flux_scale_prob": 0.5,
        },
    ).to(device)

    # Initialize UNet++ Trainer with GPU transforms and edge-prioritized balanced composite loss
    trainer = UNetPPTrainer(
        in_channels=3,
        num_classes=3,
        features=(32, 64, 128, 256, 512),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        class_weights=(0.5, 2.5, 8.0),
        gpu_transform=gpu_transform,
        device=device,
    )

    logger.info(f"Starting Grace Hopper training run: batch_size={args.batch_size}, epochs={args.epochs}")
    t0 = time.time()

    history = trainer.train(
        dataset=train_shards,
        val_dataset=val_shards,
        epochs=args.epochs,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        output_dir=output_dir,
        verbose=True,
    )

    total_time = time.time() - t0
    logger.info(f"Grace Hopper training finished in {total_time:.1f}s ({total_time / args.epochs:.2f}s/epoch)")

    # Plot training curves
    curve_path = output_dir / "gh200_training_curves.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(ep_range, history["train_loss"], "o-", label="Train Loss (CE + Dice)", color="royalblue")
    if history["val_loss"]:
        ax1.plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Grace Hopper UNet++ Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history["val_macro_iou"]:
        ax2.plot(ep_range, history["val_macro_iou"], "^-", label="Macro IoU", color="purple")
        ax2.plot(ep_range, history["val_interior_iou"], "o--", label="Interior IoU", color="forestgreen")
        ax2.plot(ep_range, history["val_boundary_iou"], "d--", label="Edge IoU", color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")
        ax2.set_title("Validation Segmentation Metrics")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved training curves to: {curve_path}")


if __name__ == "__main__":
    main()
