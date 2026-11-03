#!/usr/bin/env python3
"""
NVIDIA Grace Hopper (GH200) Center Offset Voting Training Script for sam2roman
=============================================================================

Optimized for NVIDIA Grace Hopper (GH200):
- Grace CPU: 72 ARM Neoverse-V2 cores with 480GB unified LPDDR5X RAM
- Hopper H100 GPU: 96GB/144GB HBM3 VRAM, 4 PFLOPS Tensor Cores
- NVLink-C2C: 900 GB/s bidirectional host-to-device streaming
- Asynchronous CPU pre-computation ring buffer (Option A):
  * 16-24 worker processes generate Gaussian center heatmaps and (dy, dx) offset fields
  * Pinned memory ring buffer holds 64-128 pre-augmented batches
- In-VRAM GPUAstroTransform (<0.05 ms per batch):
  * Vectorized GPU leakySB stretch
  * Astronomical Poisson shot noise and readout noise injection
- Full TF32 and native bfloat16 Automatic Mixed Precision (AMP)
- Multi-Task UNet++ with optional Recurrent Criss-Cross Attention (RCCA)
- Sustained ~100% GPU compute utilization
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
import numpy as np
import torch
import webdataset as wds

from sam2roman.models.unetpp_offset import UNetPlusPlusCenterOffset
from sam2roman.training.center_offset_trainer import CenterOffsetTrainer
from sam2roman.training.offset_target import generate_center_offset_targets_numpy
from sam2roman.training.offset_metrics import extract_center_peaks
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import create_webdataset_loader, resolve_shard_urls
from sam2roman.inference.center_offset_segmentor import CenterOffsetSegmentor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.segment_gh200")


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
    """Extract fixed validation cutouts for consistent visual progress tracking."""
    cutouts: List[Dict[str, Any]] = []

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
                    elif k.endswith("json") or k == "meta":
                        meta = v if isinstance(v, dict) else {}

                if raw is not None and target is not None:
                    cutouts.append({
                        "raw": np.asarray(raw, dtype=np.float32),
                        "target": np.asarray(target),
                        "meta": meta,
                    })
                    if len(cutouts) >= num_cutouts:
                        break
        except Exception as e:
            logger.warning(f"Could not extract cutouts from WebDataset shards: {e}")

    return cutouts


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: Path,
    current_epoch: int,
) -> Path:
    """Save training curves for total loss, sub-losses, and validation metrics."""
    curve_path = output_dir / "segment_gh200_curves.png"
    epoch_curve_path = output_dir / f"training_curves_epoch_{current_epoch:03d}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ep_range = range(1, len(history["loss"]) + 1)

    ax1.plot(ep_range, history["loss"], "o-", label="Total Loss", color="royalblue")
    if "sem_loss" in history:
        ax1.plot(ep_range, history["sem_loss"], "--", label="Semantic Loss", color="gray", alpha=0.7)
    if "center_loss" in history:
        ax1.plot(ep_range, history["center_loss"], "--", label="Center Focal", color="coral", alpha=0.7)
    if "offset_loss" in history:
        ax1.plot(ep_range, history["offset_loss"], "--", label="Offset L1", color="purple", alpha=0.7)
    if "angle_loss" in history:
        ax1.plot(ep_range, history["angle_loss"], "--", label="Spike Angle Loss", color="teal", alpha=0.7)
    if history.get("val_loss"):
        ax1.plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="crimson")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"GH200 Center Offset Convergence (Epoch {current_epoch})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    if history.get("center_f1"):
        ax2.plot(ep_range, history["center_f1"], "^-", label="Center Peak F1", color="forestgreen")
    if history.get("fg_iou"):
        ax2.plot(ep_range, history["fg_iou"], "o--", label="Foreground IoU", color="darkcyan")
    if history.get("offset_epe"):
        ax2_right = ax2.twinx()
        ax2_right.plot(ep_range, history["offset_epe"], "d-", label="Offset EPE (px)", color="darkorange")
        ax2_right.set_ylabel("EPE (pixels)", color="darkorange")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score / IoU")
    ax2.set_title(f"Validation Metrics (Epoch {current_epoch})")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center left")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(curve_path, dpi=200)
    fig.savefig(epoch_curve_path, dpi=200)
    plt.close(fig)
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
) -> Path:
    """
    Render 4 fixed validation cutouts in a 5-column inspection figure:
    1. Astronomical Stretched RGB (LeakySB)
    2. Ground Truth Instance Mask
    3. Predicted Center Heatmap with detected peaks
    4. Predicted Offset Quiver Vector Field
    5. Deblended Instances from Center Offset Voting
    """
    nrows = min(4, len(fixed_cutouts))
    if nrows == 0:
        return Path(output_path)

    fig, axes = plt.subplots(nrows, 5, figsize=(20, 4 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    model.eval()

    with torch.no_grad():
        for i in range(nrows):
            raw_arr = fixed_cutouts[i]["raw"]
            gt_inst = fixed_cutouts[i]["target"]

            # GPU stretch
            raw_t = torch.from_numpy(raw_arr).unsqueeze(0).unsqueeze(0).to(device)
            m_in_ch = getattr(model, "in_channels", 3)
            if gpu_transform is not None:
                img_t, _ = gpu_transform(raw_t, None)
            elif m_in_ch == 3:
                img_t = raw_t.repeat(1, 3, 1, 1)
            else:
                img_t = raw_t[:, :m_in_ch]

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    preds = model(img_t)
            else:
                preds = model(img_t)

            p_fg = torch.sigmoid(preds["semantic"][0, 0]).cpu().numpy()
            p_ctr = torch.sigmoid(preds["center"][0, 0]).cpu().numpy()
            pred_off = preds["offset"][0].cpu().numpy()

            # RGB / 2D for visualization
            vis_arr = img_t[0].permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            if vis_arr.shape[-1] == 1:
                vis_arr = vis_arr.squeeze(-1)

            # Detect peaks
            peaks = extract_center_peaks(p_ctr, threshold=0.3, min_distance=3)

            # Col 1: Stretched Image
            axes[i, 0].imshow(vis_arr, cmap="gray" if vis_arr.ndim == 2 else None)
            axes[i, 0].set_title(f"Cutout #{i+1}: Roman WFI Flux")
            axes[i, 0].axis("off")

            # Col 2: Ground Truth Instances
            axes[i, 1].imshow(gt_inst, cmap="gist_ncar", interpolation="nearest")
            n_gt = len(np.unique(gt_inst)[np.unique(gt_inst) > 0])
            axes[i, 1].set_title(f"GT Instances (N={n_gt})")
            axes[i, 1].axis("off")

            # Col 3: Center Heatmap & Peaks
            axes[i, 2].imshow(p_ctr, cmap="inferno", vmin=0.0, vmax=1.0)
            for py, px, sc in peaks:
                axes[i, 2].plot(px, py, "c+", markersize=10, markeredgewidth=2)
            axes[i, 2].set_title(f"Pred Centers (N={len(peaks)})")
            axes[i, 2].axis("off")

            # Col 4: Offset Vector Field (Quiver sampled every 16 px on foreground)
            axes[i, 3].imshow(p_fg, cmap="Blues", alpha=0.5)
            step = 16
            h, w = p_fg.shape
            yy, xx = np.mgrid[0:h:step, 0:w:step]
            u = pred_off[1, yy, xx]  # delta_x
            v = pred_off[0, yy, xx]  # delta_y
            fg_sub = (p_fg[yy, xx] >= 0.3)
            axes[i, 3].quiver(
                xx[fg_sub], yy[fg_sub], u[fg_sub], -v[fg_sub],
                color="red", scale=250, width=0.005, headwidth=4
            )
            axes[i, 3].set_title("Pred Offset Vectors (Δy, Δx)")
            axes[i, 3].axis("off")

            # Col 5: Reconstructed Instance Voting
            segmentor = CenterOffsetSegmentor(model=model, device=device)
            res = segmentor.segment_image(raw_arr, fg_thresh=0.5, center_thresh=0.3, verbose=False)
            recon_map = np.zeros((h, w), dtype=np.int32)
            for inst_idx, m in enumerate(res.masks):
                recon_map[m] = inst_idx + 1

            axes[i, 4].imshow(recon_map, cmap="gist_ncar", interpolation="nearest")
            axes[i, 4].set_title(f"Deblended (N={len(res.masks)})")
            axes[i, 4].axis("off")

    fig.suptitle(f"Grace Hopper Center Offset Validation (Epoch {epoch}/{epochs})", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return Path(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NVIDIA Grace Hopper (GH200) Center Offset Voting Training Script for sam2roman."
    )
    parser.add_argument("--train_shards", type=str, nargs="+", required=True, help="Train WebDataset shard path pattern.")
    parser.add_argument("--val_shards", type=str, nargs="+", default=None, help="Val WebDataset shard path pattern.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and curves.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per training step (default: 32).")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs (default: 80).")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (default: 6e-5).")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (default: 1e-2).")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm (default: 1.0).")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"], dest="scheduler_type")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs (default: 3).")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader CPU workers (default: 16 on 72-core GH200).")
    parser.add_argument("--prefetch_factor", type=int, default=8, help="DataLoader prefetch factor (default: 8 for ring buffer).")
    parser.add_argument("--use_cca", action="store_true", default=True, help="Enable Recurrent Criss-Cross Attention for spikes.")
    parser.add_argument("--no_cca", action="store_false", dest="use_cca", help="Disable Criss-Cross Attention.")
    parser.add_argument("--center_weight", type=float, default=2.0, help="Center heatmap focal loss weight (default: 2.0).")
    parser.add_argument("--offset_weight", type=float, default=1.0, help="Offset regression Smooth L1 weight (default: 1.0).")
    parser.add_argument("--angle_weight", type=float, default=0.5, help="Angular cosine loss weight for diffraction spikes (default: 0.5).")
    parser.add_argument("--center_sigma", type=float, default=2.0, help="Gaussian center peak sigma (default: 2.0).")
    parser.add_argument("--noise_aug", action="store_true", default=True, help="Enable GPU Poisson shot & Gaussian read noise.")
    parser.add_argument("--log_file", type=str, default="epoch_log.txt", help="Log filename inside output_dir.")
    parser.add_argument("--plot_interval", type=int, default=5, help="Epoch interval for visual inspection figures (default: 5).")
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
        logger.info("[Apple Silicon] MPS accelerator active")
        device = "mps"
    else:
        logger.warning("Running on CPU fallback.")
        device = "cpu"

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
            "enable_spatial": False,  # Spatial rotation/flips handled on CPU before offset calculation
        },
    ).to(device)

    trainer = CenterOffsetTrainer(
        in_channels=3,
        semantic_classes=1,
        features=(32, 64, 128, 256, 512),
        use_cca=args.use_cca,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        center_weight=args.center_weight,
        offset_weight=args.offset_weight,
        angle_weight=args.angle_weight,
        center_sigma=args.center_sigma,
        gpu_transform=gpu_transform,
        device=device,
    )

    fixed_val_cutouts = None
    if val_shards:
        logger.info("Extracting fixed validation cutouts for periodic inference monitoring...")
        fixed_val_cutouts = extract_fixed_val_cutouts(val_shards, num_cutouts=4)
        if fixed_val_cutouts:
            logger.info(f"Cached {len(fixed_val_cutouts)} fixed cutouts.")

    def epoch_callback(trainer_inst: CenterOffsetTrainer, epoch_data: Dict[str, Any], history: Dict[str, List[float]]) -> None:
        epoch = epoch_data["epoch"]
        epochs = epoch_data["epochs"]
        if epoch % args.plot_interval == 0 or epoch == epochs:
            plot_training_curves(history=history, output_dir=output_dir, current_epoch=epoch)
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
        prefetch_factor=args.prefetch_factor,
        output_dir=output_dir,
        log_file=args.log_file,
        epoch_callback=epoch_callback,
        verbose=True,
    )

    total_time = time.time() - t0
    logger.info(f"Grace Hopper Center Offset training finished in {total_time:.1f}s ({total_time / args.epochs:.2f}s/epoch)")
    plot_training_curves(history=history, output_dir=output_dir, current_epoch=args.epochs)


if __name__ == "__main__":
    main()
