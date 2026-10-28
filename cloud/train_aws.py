#!/usr/bin/env python3
"""
AWS GPU Training Script for sam2roman
======================================
Optimized for AWS EC2 instances with 24GB NVIDIA GPUs:
- g6.xlarge / g6.2xlarge: NVIDIA L4 GPU (24GB GDDR6, Ada Lovelace, native bfloat16)
- g5.xlarge / g5.2xlarge: NVIDIA A10G GPU (24GB GDDR6, Ampere)

Features:
- S3 dataset synchronization (s3:// URI support)
- Multi-mask crop batching (trains on ALL sources in each 512x512 crop simultaneously)
- Default leakySB surface brightness stretch (identical RGB channels)
- Automatic mixed precision (AMP with native bfloat16/fp16)
- Checkpoint management and training curve generation
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
import numpy as np
import torch

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.trainer import SAM2Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.aws")


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


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    """Upload local directory or file to an S3 URI."""
    logger.info(f"Uploading artifacts to S3 '{s3_uri}'...")
    if local_path.is_file():
        cmd = ["aws", "s3", "cp", str(local_path), s3_uri]
    else:
        cmd = ["aws", "s3", "sync", str(local_path), s3_uri, "--exclude", "*.DS_Store"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upload to S3: {e}")


def plot_training_history(history: dict, output_path: Path) -> None:
    """Save training and validation loss/IoU curves to a PNG image."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    has_val = bool(history.get("val_loss") and len(history["val_loss"]) == len(epochs))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    if has_val:
        ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Segmentation Loss (BCE + Dice)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    if has_val and "val_iou" in history:
        ax2 = ax1.twinx()
        ax2.plot(epochs, history["val_iou"], "g-.^", label="Val IoU", linewidth=2)
        ax2.set_ylabel("Validation IoU", color="g", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="g")

    fig.tight_layout()
    plt.title("SAM 2 Fine-Tuning Curves (AWS GPU)", fontsize=14, pad=12)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved training curve plot: {output_path}")


def print_gpu_diagnostics():
    """Print hardware information for AWS GPU instance."""
    print("=" * 75)
    print("              sam2roman: AWS Cloud Training Engine                     ")
    print("=" * 75)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"  -> Hardware:     {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  -> CUDA Version: {torch.version.cuda}")
        print(f"  -> Mixed Prec:   bfloat16 {'(Supported & Enabled)' if bf16_ok else '(fp16 Enabled)'}")
        if vram_gb >= 20.0:
            print("  -> Configuration: 24GB GPU detected (Optimal batch_size: 8 - 16)")
    else:
        print("  -> Hardware:     CPU (No CUDA device found)")
    print("-" * 75)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM 2 on astronomical FITS imagery on AWS EC2 (g6.xlarge / g5.xlarge)."
    )
    # Data arguments
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Local directory or s3:// URI containing original astronomical FITS files.",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        required=True,
        help="Local directory or s3:// URI containing ground-truth mask FITS files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_aws",
        help="Local directory or s3:// URI to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.fits",
        help="File pattern to match (default: '*.fits').",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of FITS files to match.",
    )

    # Patch / Cutout arguments
    parser.add_argument(
        "--crop_size",
        type=int,
        default=512,
        help="Cutout crop size in pixels (default: 512).",
    )
    parser.add_argument(
        "--samples_per_file",
        type=str,
        default="all",
        help="Cutouts per file ('all' or integer K like 10, 20).",
    )
    parser.add_argument(
        "--all_masks_in_crop",
        action="store_true",
        default=True,
        help="Segment ALL objects in each crop simultaneously (default: True).",
    )
    parser.add_argument(
        "--max_objects_per_crop",
        type=int,
        default=16,
        help="Cap on maximum objects per crop (default: 16).",
    )
    parser.add_argument(
        "--min_object_pixels",
        type=int,
        default=4,
        help="Minimum pixels inside crop required to train on an object (default: 4).",
    )
    parser.add_argument(
        "--stretch_type",
        type=str,
        default="leakysb",
        help="Image stretch: 'leakysb' (default), 'asinh', 'zscale', 'linear'.",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="point",
        choices=["point", "box", "both"],
        help="Training prompt mode: 'point' (default), 'box', or 'both'.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam2.1-hiera-tiny",
        help="Hugging Face SAM 2 model ID (default: 'facebook/sam2.1-hiera-tiny').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8; can use 8-16 on 24GB GPUs).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for trainable prompt encoder and mask decoder (default: 1e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation (default: 0.2).",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print_gpu_diagnostics()

    # 1. Resolve local vs S3 directories
    scratch_dir = Path("/tmp/sam2roman_data")
    if args.images_dir.startswith("s3://"):
        local_images_dir = sync_s3_path(args.images_dir, scratch_dir / "images")
    else:
        local_images_dir = Path(args.images_dir)

    if args.masks_dir.startswith("s3://"):
        local_masks_dir = sync_s3_path(args.masks_dir, scratch_dir / "masks")
    else:
        local_masks_dir = Path(args.masks_dir)

    # Local output directory for checkpoints
    s3_output_uri = args.output_dir if args.output_dir.startswith("s3://") else None
    local_output_dir = Path("/tmp/sam2roman_checkpoints") if s3_output_uri else Path(args.output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)

    samples_per_file = None if args.samples_per_file in ("all", "none", "None") else int(args.samples_per_file)

    # 2. Build AstronomicalDataset
    logger.info("Initializing AstronomicalDataset...")
    dataset = AstronomicalDataset.from_directories(
        images_dir=local_images_dir,
        masks_dir=local_masks_dir,
        pattern=args.pattern,
        crop_size=args.crop_size,
        samples_per_file=samples_per_file,
        all_masks_in_crop=args.all_masks_in_crop,
        max_objects_per_crop=args.max_objects_per_crop,
        min_object_pixels=args.min_object_pixels,
        prompt_type=args.prompt_type,
        stretch_type=args.stretch_type,
        max_samples=args.max_files,
    )
    logger.info(f"Loaded {len(dataset)} training patches (all_masks_in_crop={args.all_masks_in_crop}).")

    # 3. Train / Validation Split
    if args.val_split > 0.0 and len(dataset) > 1:
        n_val = max(1, int(len(dataset) * args.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
        logger.info(f"Dataset split: {len(train_ds)} train crops | {len(val_ds)} val crops")
    else:
        train_ds = dataset
        val_ds = None

    # 4. Initialize SAM2Trainer
    logger.info(f"Configuring SAM2Trainer with model '{args.model_id}'...")
    trainer = SAM2Trainer(
        model_id=args.model_id,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        freeze_vision_encoder=True,
        prompt_type=args.prompt_type,
        use_amp=not args.no_amp,
    )

    # 5. Execute Fine-Tuning Loop
    t_start = time.time()
    history = trainer.train(
        dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=local_output_dir,
        verbose=True,
    )
    elapsed_min = (time.time() - t_start) / 60.0

    # 6. Save metrics and plots
    metrics_path = local_output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model_id": args.model_id,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "elapsed_minutes": round(elapsed_min, 2),
                "history": history,
            },
            f,
            indent=2,
        )

    curve_path = local_output_dir / "training_curves.png"
    plot_training_history(history, curve_path)

    # 7. Upload to S3 if requested
    if s3_output_uri:
        upload_to_s3(local_output_dir, s3_output_uri)

    logger.info(f"Fine-tuning complete in {elapsed_min:.2f} minutes!")
    logger.info(f"Checkpoints and artifacts saved at: {s3_output_uri or local_output_dir}")


if __name__ == "__main__":
    main()
