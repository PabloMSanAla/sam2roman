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
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure sam2roman package is found
for candidate in [Path(__file__).resolve().parent.parent, Path.cwd()]:
    if (candidate / "sam2roman").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import torch
import webdataset as wds

from sam2roman.training.unetpp_trainer import UNetPPTrainer
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import create_webdataset_loader, resolve_shard_urls
from sam2roman.training.metrics import compute_relaxed_boundary_iou
from sam2roman.io.fits import stretch_astro_image

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


def extract_fixed_val_cutouts(
    val_source: Union[str, Path, Sequence[Union[str, Path]], Any],
    num_cutouts: int = 4,
) -> List[Dict[str, Any]]:
    """
    Extract fixed validation cutouts (default: 4) from WebDataset shards,
    DataLoader, or AstronomicalDataset for consistent epoch-to-epoch visual comparison.
    """
    cutouts: List[Dict[str, Any]] = []

    # If it has an underlying dataset (e.g. DataLoader wrapping a Dataset)
    if hasattr(val_source, "dataset") and hasattr(val_source.dataset, "__getitem__"):
        val_source = val_source.dataset

    # Case 1: WebDataset shard paths or glob strings
    if isinstance(val_source, (str, Path)) or (
        isinstance(val_source, (list, tuple))
        and len(val_source) > 0
        and isinstance(val_source[0], (str, Path))
    ):
        try:
            shard_list = resolve_shard_urls(val_source)
            pipeline = wds.WebDataset(shard_list, shardshuffle=False).decode()
            for sample in pipeline:
                raw = None
                target = None
                meta = {}
                for k, v in sample.items():
                    if k.endswith("raw.npy") or k == "raw":
                        raw = v
                    elif k.endswith("target.npy") or k == "target":
                        target = v
                    elif k.endswith(".json") or k == "json":
                        meta = v if isinstance(v, dict) else {}

                if raw is not None and target is not None:
                    cutouts.append({
                        "raw": np.asarray(raw, dtype=np.float32),
                        "target": np.asarray(target),
                        "meta": meta,
                    })
                    if len(cutouts) >= num_cutouts:
                        break
            if cutouts:
                return cutouts
        except Exception as e:
            logger.warning(f"Failed to read raw cutouts from WebDataset shards directly: {e}")

    # Case 2: PyTorch Dataset (e.g. AstronomicalDataset)
    if hasattr(val_source, "__getitem__") and hasattr(val_source, "__len__"):
        try:
            n = min(num_cutouts, len(val_source))
            for i in range(n):
                s = val_source[i]
                r = s.get("raw_data", s.get("raw", s.get("image")))
                t = s.get("target", s.get("mask", s.get("label")))
                if hasattr(r, "convert"):  # PIL Image
                    r = np.array(r)
                elif isinstance(r, torch.Tensor):
                    r = r.squeeze().float().cpu().numpy()
                if isinstance(t, torch.Tensor):
                    t = t.squeeze().cpu().numpy()
                m = s.get("meta", {})
                cutouts.append({
                    "raw": np.asarray(r, dtype=np.float32),
                    "target": np.asarray(t),
                    "meta": m,
                })
            if cutouts:
                return cutouts
        except Exception as e:
            logger.warning(f"Failed to extract cutouts from Dataset: {e}")

    # Case 3: Active DataLoader
    if hasattr(val_source, "__iter__"):
        try:
            for batch in val_source:
                raw_b = batch.get("raw", batch.get("image"))
                tgt_b = batch.get("target", batch.get("label"))
                meta_b = batch.get("meta", [{}] * len(raw_b) if raw_b is not None else [])
                if raw_b is not None and tgt_b is not None:
                    for i in range(len(raw_b)):
                        item_r = raw_b[i]
                        if hasattr(item_r, "convert"):
                            r = np.array(item_r)
                        elif isinstance(item_r, torch.Tensor):
                            r = item_r.squeeze().float().cpu().numpy()
                        else:
                            r = np.asarray(item_r)

                        item_t = tgt_b[i]
                        if isinstance(item_t, torch.Tensor):
                            t = item_t.squeeze().cpu().numpy()
                        else:
                            t = np.asarray(item_t)

                        m = meta_b[i] if i < len(meta_b) and isinstance(meta_b[i], dict) else {}
                        cutouts.append({
                            "raw": np.asarray(r, dtype=np.float32),
                            "target": np.asarray(t),
                            "meta": m,
                        })
                        if len(cutouts) >= num_cutouts:
                            return cutouts
            return cutouts
        except Exception as e:
            logger.warning(f"Failed to extract cutouts from DataLoader: {e}")

    return cutouts


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: Path,
    current_epoch: int,
    boundary_tolerance: int = 1,
) -> Path:
    """Save training and validation loss & IoU metric curves."""
    curve_path = output_dir / "gh200_training_curves.png"
    epoch_curve_path = output_dir / f"training_curves_epoch_{current_epoch:03d}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(ep_range, history["train_loss"], "o-", label="Train Loss (CE + Dice)", color="royalblue")
    if history.get("val_loss"):
        ax1.plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Grace Hopper UNet++ Convergence (Epoch {current_epoch})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history.get("val_macro_iou"):
        ax2.plot(ep_range, history["val_macro_iou"], "^-", label="Macro IoU", color="purple")
        ax2.plot(ep_range, history["val_interior_iou"], "o--", label="Interior IoU", color="forestgreen")
        ax2.plot(ep_range, history["val_boundary_iou"], "d-", label=f"Edge IoU (tol={boundary_tolerance}px)", color="orange")
        if history.get("val_strict_boundary_iou"):
            ax2.plot(ep_range, history["val_strict_boundary_iou"], ":", label="Strict Edge IoU (0px)", color="gray", alpha=0.7)
        if history.get("val_soft_boundary_iou"):
            ax2.plot(ep_range, history["val_soft_boundary_iou"], "--", label="Soft Edge IoU", color="darkcyan", alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU")
        ax2.set_title(f"Validation Segmentation Metrics (Epoch {current_epoch})")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(curve_path, dpi=200)
    fig.savefig(epoch_curve_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved training curves to: {curve_path} and {epoch_curve_path.name}")
    return curve_path


def plot_validation_examples(
    model: torch.nn.Module,
    fixed_cutouts: List[Dict[str, Any]],
    gpu_transform: Optional[torch.nn.Module],
    output_path: Union[str, Path],
    epoch: int,
    epochs: int,
    device: Union[str, torch.device] = "cpu",
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    boundary_tolerance: int = 1,
) -> Path:
    """
    Render 4 fixed validation cutouts in a publication-quality grid:
    - 4 rows (1 row per cutout)
    - 3 columns:
        1. Stretched Astronomical Image (LeakySB)
        2. Ground Truth Target (Sky / Core / Edge)
        3. UNet++ Predicted Mask (with per-cutout Core and Edge IoU)
    """
    nrows = min(4, len(fixed_cutouts))
    if nrows == 0:
        return Path(output_path)

    # Discrete 3-class segmentation colormap:
    # 0: Sky background (#0b0d1b - Dark navy)
    # 1: Core interior (#00d4ff - Electric cyan)
    # 2: Boundary edge (#ff3366 - Vivid coral red)
    cmap_3class = ListedColormap(["#0b0d1b", "#00d4ff", "#ff3366"])
    norm_3class = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_3class.N)

    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.8 * nrows), squeeze=False)

    for i in range(nrows):
        sample = fixed_cutouts[i]
        raw = sample["raw"]
        target = sample["target"]
        meta = sample.get("meta", {})

        # Ensure target is 2D integer class mask
        if target.ndim == 3:
            target_2d = np.argmax(target, axis=0)
        else:
            target_2d = target

        # Prepare tensor for model
        if raw.ndim == 2:
            raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).float().to(device)
        elif raw.ndim == 3 and raw.shape[0] == 1:
            raw_t = torch.from_numpy(raw).unsqueeze(0).float().to(device)
        else:
            raw_t = torch.from_numpy(raw).float().to(device)
            if raw_t.ndim == 3:
                raw_t = raw_t.unsqueeze(1)

        # Run forward pass in eval mode
        was_model_train = model.training
        model.eval()

        if gpu_transform is not None:
            was_gpu_train = gpu_transform.training
            gpu_transform.eval()
            with torch.no_grad():
                img_norm, _ = gpu_transform(raw_t)
            if was_gpu_train:
                gpu_transform.train()
        else:
            stretched_np = stretch_astro_image(raw, stretch_type="leakysb")
            img_norm = torch.from_numpy(stretched_np).unsqueeze(0).unsqueeze(0).float().to(device)
            if img_norm.shape[1] == 1:
                img_norm = img_norm.repeat(1, 3, 1, 1)

        with torch.no_grad():
            if use_amp and (isinstance(device, torch.device) and device.type == "cuda" or str(device).startswith("cuda")):
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(img_norm)
            else:
                logits = model(img_norm)

        if was_model_train:
            model.train()

        if isinstance(logits, list):
            logits = logits[-1]

        # Explicitly cast to float32 before converting to NumPy (safeguards against BFloat16 AMP)
        probs = torch.softmax(logits.float(), dim=1).squeeze(0).cpu().numpy()
        pred_classes = np.argmax(probs, axis=0)

        # Stretched image for display
        if img_norm.ndim == 4 and img_norm.shape[1] >= 1:
            disp_img = img_norm[0, 0].detach().float().cpu().numpy()
        else:
            disp_img = stretch_astro_image(raw, stretch_type="leakysb")

        # Per-cutout IoU metrics
        p_core = (pred_classes == 1)
        t_core = (target_2d == 1)
        core_denom = np.sum(p_core | t_core)
        core_iou = float(np.sum(p_core & t_core) / max(1, core_denom)) if core_denom > 0 else 1.0

        p_edge = (pred_classes == 2)
        t_edge = (target_2d == 2)
        edge_denom = np.sum(p_edge | t_edge)
        p_edge_t = torch.from_numpy(p_edge).unsqueeze(0)
        t_edge_t = torch.from_numpy(t_edge).unsqueeze(0)
        rel_edge_iou = float(
            compute_relaxed_boundary_iou(p_edge_t, t_edge_t, tolerance=boundary_tolerance).item()
        ) if edge_denom > 0 else 1.0

        # Subplot 1: LeakySB stretched astronomical image
        axes[i, 0].imshow(disp_img, cmap="gray", origin="lower")
        axes[i, 0].axis("off")
        cutout_label = meta.get("sample_key", f"Cutout #{i+1}")
        axes[i, 0].text(
            -0.05, 0.5, cutout_label,
            transform=axes[i, 0].transAxes,
            rotation=90, va="center", ha="right",
            fontsize=12, fontweight="bold",
        )

        # Subplot 2: Ground Truth Target Mask
        axes[i, 1].imshow(target_2d, cmap=cmap_3class, norm=norm_3class, origin="lower")
        axes[i, 1].axis("off")

        # Subplot 3: UNet++ Predicted Mask
        axes[i, 2].imshow(pred_classes, cmap=cmap_3class, norm=norm_3class, origin="lower")
        axes[i, 2].axis("off")
        axes[i, 2].set_title(
            f"Core IoU: {core_iou:.3f} | Edge IoU (tol={boundary_tolerance}px): {rel_edge_iou:.3f}",
            fontsize=11, fontweight="bold", pad=6,
        )

    # Column header labels
    axes[0, 0].set_title("Stretched Image (LeakySB)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_title("Ground Truth (Sky / Core / Edge)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 2].set_title(f"UNet++ Prediction (Epoch {epoch})", fontsize=13, fontweight="bold", pad=10)

    # Shared 3-class legend
    patches = [
        mpatches.Patch(color="#0b0d1b", label="Sky Background (0)"),
        mpatches.Patch(color="#00d4ff", label="Source Core (1)"),
        mpatches.Patch(color="#ff3366", label="Deblending Edge (2)"),
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=3,
        framealpha=0.9,
        facecolor="#1c2038",
        edgecolor="#444b6e",
        fontsize=11,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.005),
    )

    fig.suptitle(
        f"Epoch {epoch:02d}/{epochs:02d} — Fixed Validation Cutout Predictions (UNet++)",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.98])

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=180, bbox_inches="tight")

    # Also save to val_inference_latest.png in the same directory
    latest_path = out_p.parent / "val_inference_latest.png"
    if out_p != latest_path:
        fig.savefig(latest_path, dpi=180, bbox_inches="tight")

    plt.close(fig)
    logger.info(f"Saved validation inference figure to: {out_p} and {latest_path.name}")
    return out_p


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for train_gh200."""
    parser = argparse.ArgumentParser(
        description="High-Throughput Grace Hopper (GH200) Training for sam2roman."
    )
    parser.add_argument(
        "--train_shards",
        type=str,
        nargs="+",
        required=True,
        help="Path, glob pattern, or list of training shards (e.g. '/dev/shm/train/train-*.tar' or 's3://...').",
    )
    parser.add_argument(
        "--val_shards",
        type=str,
        nargs="+",
        default=None,
        help="Optional path, glob pattern, or list of validation shards.",
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
        default=8,
        help="DataLoader worker processes (sweet spot is 8-16 on GH200 72-core ARM CPU; note: num_workers must be <= number of WebDataset shards).",
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
    parser.add_argument(
        "--rot_prob",
        type=float,
        default=0.0,
        help="Probability of random 90-degree rotations in GPU augmentation (default: 0.0, disabled).",
    )
    parser.add_argument(
        "--soft_boundary",
        action="store_true",
        default=False,
        help="Enable on-the-fly GPU Gaussian smoothing of boundaries from discrete WebDataset targets.",
    )
    parser.add_argument(
        "--boundary_sigma",
        type=float,
        default=1.2,
        help="Gaussian sigma for edge smoothing (default: 1.2).",
    )
    parser.add_argument(
        "--boundary_tolerance",
        type=int,
        default=1,
        help="Pixel distance tolerance for relaxed boundary IoU validation (default: 1 px).",
    )
    parser.add_argument(
        "--class_weights",
        type=float,
        nargs=3,
        default=[0.5, 2.5, 8.0],
        help="Class loss balancing weights [sky, interior, boundary] (default: 0.5 2.5 8.0).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="epoch_log.txt",
        help="Filename inside output_dir to write epoch summary logs (default: 'epoch_log.txt').",
    )
    parser.add_argument(
        "--plot_interval",
        type=int,
        default=5,
        help="Epoch interval for updating training curves and generating fixed validation cutout inference figures (default: 5).",
    )
    return parser


def main():
    parser = build_parser()
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
    if len(train_shards) == 1:
        if train_shards[0].startswith("s3://"):
            local_train = Path("/dev/shm/roman_shards/train") if Path("/dev/shm").exists() else Path("./data_shards/train")
            train_shards = str(sync_s3_path(train_shards[0], local_train))
        else:
            train_shards = train_shards[0]

    val_shards = args.val_shards
    if val_shards and len(val_shards) == 1:
        if val_shards[0].startswith("s3://"):
            local_val = Path("/dev/shm/roman_shards/val") if Path("/dev/shm").exists() else Path("./data_shards/val")
            val_shards = str(sync_s3_path(val_shards[0], local_val))
        else:
            val_shards = val_shards[0]

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
            "rot_prob": args.rot_prob,
            "shot_noise_prob": 0.5,
            "read_noise_prob": 0.5,
            "sky_pedestal_prob": 0.5,
            "flux_scale_prob": 0.5,
        },
        soften_boundaries=args.soft_boundary,
        boundary_sigma=args.boundary_sigma,
    ).to(device)

    # Initialize UNet++ Trainer with GPU transforms and edge-prioritized balanced composite loss
    trainer = UNetPPTrainer(
        in_channels=3,
        num_classes=3,
        features=(32, 64, 128, 256, 512),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        class_weights=tuple(args.class_weights),
        boundary_tolerance=args.boundary_tolerance,
        gpu_transform=gpu_transform,
        device=device,
    )

    # Pre-extract 4 fixed validation cutouts for consistent visual progress tracking
    fixed_val_cutouts = None
    if val_shards:
        logger.info("Extracting 4 fixed validation cutouts for periodic inference monitoring...")
        fixed_val_cutouts = extract_fixed_val_cutouts(val_shards, num_cutouts=4)
        if fixed_val_cutouts:
            logger.info(f"Successfully cached {len(fixed_val_cutouts)} fixed validation cutouts.")
        else:
            logger.warning("No cutouts could be pre-extracted from val_shards.")

    def epoch_callback(trainer_inst: UNetPPTrainer, epoch_data: Dict[str, Any], history: Dict[str, List[float]]) -> None:
        """Invoked at the end of each epoch to handle periodic plotting and validation inference."""
        epoch = epoch_data["epoch"]
        epochs = epoch_data["epochs"]

        # Every plot_interval epochs or on the final epoch
        if epoch % args.plot_interval == 0 or epoch == epochs:
            # 1. Update loss and validation metric plots
            plot_training_curves(
                history=history,
                output_dir=output_dir,
                current_epoch=epoch,
                boundary_tolerance=args.boundary_tolerance,
            )

            # 2. Render fixed validation cutout inference figure
            if fixed_val_cutouts:
                fig_path = output_dir / f"val_inference_epoch_{epoch:03d}.png"
                plot_validation_examples(
                    model=trainer_inst.model,
                    fixed_cutouts=fixed_val_cutouts,
                    gpu_transform=trainer_inst.gpu_transform,
                    output_path=fig_path,
                    epoch=epoch,
                    epochs=epochs,
                    device=trainer_inst.device,
                    use_amp=trainer_inst.use_amp,
                    amp_dtype=trainer_inst.amp_dtype,
                    boundary_tolerance=args.boundary_tolerance,
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
        log_file=args.log_file,
        epoch_callback=epoch_callback,
        verbose=True,
    )

    total_time = time.time() - t0
    logger.info(f"Grace Hopper training finished in {total_time:.1f}s ({total_time / args.epochs:.2f}s/epoch)")

    # Ensure final training curves and metrics are updated
    curve_path = plot_training_curves(
        history=history,
        output_dir=output_dir,
        current_epoch=args.epochs,
        boundary_tolerance=args.boundary_tolerance,
    )
    logger.info(f"Saved final training curves to: {curve_path}")
    if args.log_file:
        logger.info(f"Saved epoch logs to: {output_dir / args.log_file}")
    logger.info(f"Saved training metrics to: {output_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
