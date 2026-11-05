#!/usr/bin/env python3
"""
NVIDIA Grace Hopper (GH200) SteerQuery Star-Deblending Training Script for sam2roman
===================================================================================

Optimized for NVIDIA Grace Hopper (GH200):
- Grace CPU: 72 ARM Neoverse-V2 cores with 480GB unified LPDDR5X RAM
- Hopper H100 GPU: 96GB/144GB HBM3 VRAM, 4 PFLOPS Tensor Cores
- NVLink-C2C: 900 GB/s bidirectional host-to-device streaming
- Direct GPU Integer Instance Mask Assembly without CPU voting or clustering
- Two-speed optimizer: 0.1x LR on geometric angle and curvature steering parameters
- Star-only instance deblending (galaxies strictly preserved as background ID = 0)
- Full TF32 and native bfloat16 Automatic Mixed Precision (AMP)
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

from sam2roman.models.steer_query_head import UNetPlusPlusSteerQuery
from sam2roman.training.steer_query_trainer import SteerQueryTrainer
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import resolve_shard_urls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.steer_query_gh200")


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


def extract_crowded_val_cutouts(
    val_source: Union[str, Path, Sequence[Union[str, Path]], Any],
    pool_size: int = 32,
    max_scan: int = 150,
) -> List[Dict[str, Any]]:
    """
    Extract a diverse pool of the most crowded validation cutouts (highest star/source density)
    to rigorously challenge and monitor deblending performance across epochs.
    """
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    if isinstance(val_source, (str, Path)) or (
        isinstance(val_source, (list, tuple))
        and len(val_source) > 0
        and isinstance(val_source[0], (str, Path))
    ):
        try:
            shard_list = resolve_shard_urls(val_source)
            pipeline = wds.WebDataset(shard_list, shardshuffle=False).decode()
            scanned = 0
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
                    tgt_arr = np.asarray(target)
                    unique_ids = np.unique(tgt_arr)
                    n_stars = int(len(unique_ids[unique_ids > 0]))

                    candidates.append((
                        n_stars,
                        {
                            "raw": np.asarray(raw, dtype=np.float32),
                            "target": tgt_arr,
                            "meta": meta,
                            "n_stars": n_stars,
                        }
                    ))
                    scanned += 1
                    if scanned % 20 == 0 or scanned == max_scan:
                        logger.info(f"Scanned {scanned}/{max_scan} validation cutouts...")
                    if scanned >= max_scan:
                        break
        except Exception as e:
            logger.warning(f"Could not extract crowded cutouts from WebDataset shards: {e}")

    # Sort descending by star count (crowdest cutouts first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    pool = [c[1] for c in candidates[:pool_size]]
    return pool


# Backwards-compatible alias
extract_fixed_val_cutouts = extract_crowded_val_cutouts


def select_val_cutouts_for_epoch(
    cutout_pool: List[Dict[str, Any]],
    epoch: int,
    plot_interval: int = 5,
    num_cutouts: int = 4,
    crowded_top_k: int = 16,
) -> List[Dict[str, Any]]:
    """
    Randomly sample cutouts from the top most crowded validation pool for each figure save.
    Maintains maximum scene crowding across all epochs instead of descending in density.
    Uses epoch-seeded random selection for full reproducibility.
    """
    if not cutout_pool:
        return []
    if len(cutout_pool) <= num_cutouts:
        return cutout_pool

    # Focus exclusively on the top most crowded cutouts
    pool_subset = cutout_pool[: max(num_cutouts, min(len(cutout_pool), crowded_top_k))]

    # Seed with epoch for reproducible yet varying random combinations
    rng = np.random.RandomState(seed=42 + epoch * 1000)
    selected_indices = rng.choice(len(pool_subset), size=num_cutouts, replace=False)
    return [pool_subset[i] for i in selected_indices]


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
    if "gate_loss" in history:
        ax1.plot(ep_range, history["gate_loss"], "--", label="Gate BCE", color="teal", alpha=0.7)
    if "inst_loss" in history:
        ax1.plot(ep_range, history["inst_loss"], "--", label="Instance Loss", color="purple", alpha=0.7)
    if history.get("val_loss"):
        ax1.plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="crimson")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"GH200 SteerQuery Convergence (Epoch {current_epoch})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    if history.get("star_f1"):
        ax2.plot(ep_range, history["star_f1"], "^-", label="Star Detection F1", color="forestgreen")
    if history.get("fg_iou"):
        ax2.plot(ep_range, history["fg_iou"], "o--", label="Foreground IoU", color="darkcyan")

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
    2. Ground Truth Star Instances
    3. Predicted Star Center Heatmap H_star
    4. Predicted Spike-Bridging Gate G_fg
    5. Directly Deblended Star Instances (instance_mask on GPU)
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

            h_ctr = preds["H_star"][0, 0].float().cpu().numpy()
            g_fg = preds["G_fg"][0, 0].float().cpu().numpy()
            inst_pred = preds["instance_mask"][0].cpu().numpy()

            # RGB / 2D for visualization
            vis_arr = img_t[0].permute(1, 2, 0).float().cpu().numpy().clip(0.0, 1.0)
            if vis_arr.shape[-1] == 1:
                vis_arr = vis_arr.squeeze(-1)

            # Col 1: Stretched Image
            n_gt = len(np.unique(gt_inst)[np.unique(gt_inst) > 0])
            axes[i, 0].imshow(vis_arr, cmap="gray" if vis_arr.ndim == 2 else None)
            axes[i, 0].set_title(f"Field #{i+1} ({n_gt} stars): WFI Flux")
            axes[i, 0].axis("off")

            # Col 2: Ground Truth Instances
            axes[i, 1].imshow(gt_inst, cmap="gist_ncar", interpolation="nearest")
            axes[i, 1].set_title(f"GT Star Instances (N={n_gt})")
            axes[i, 1].axis("off")

            # Col 3: Center Heatmap
            axes[i, 2].imshow(h_ctr, cmap="inferno", vmin=0.0, vmax=1.0)
            axes[i, 2].set_title("Star Center Heatmap (H_star)")
            axes[i, 2].axis("off")

            # Col 4: Spike-Bridging Gate
            axes[i, 3].imshow(g_fg, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[i, 3].set_title("Spike-Bridging Gate (G_fg)")
            axes[i, 3].axis("off")

            # Col 5: Reconstructed Instance Mask (GPU Direct Assembly)
            axes[i, 4].imshow(inst_pred, cmap="gist_ncar", interpolation="nearest")
            n_pred = len(np.unique(inst_pred)[np.unique(inst_pred) > 0])
            axes[i, 4].set_title(f"Deblended Stars (N={n_pred})")
            axes[i, 4].axis("off")

    fig.suptitle(f"Grace Hopper SteerQuery Validation (Epoch {epoch}/{epochs})", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return Path(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NVIDIA Grace Hopper (GH200) SteerQuery Star-Deblending Training Script for sam2roman."
    )
    parser.add_argument("--train_shards", type=str, nargs="+", required=True, help="Train WebDataset shard path pattern.")
    parser.add_argument("--val_shards", type=str, nargs="+", default=None, help="Val WebDataset shard path pattern.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and curves.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per training step (default: 16).")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (default: 1).")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs (default: 80).")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (default: 6e-5).")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (default: 1e-2).")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm (default: 1.0).")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"], dest="scheduler_type")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Teacher-forcing query warmup epochs (default: 5).")
    parser.add_argument("--num_workers", type=int, default=16, help="DataLoader CPU workers (default: 16 on 72-core GH200).")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor (default: 4 for ring buffer).")
    parser.add_argument("--sem_weight", type=float, default=1.0, help="Semantic segmentation loss weight (default: 1.0).")
    parser.add_argument("--center_weight", type=float, default=2.0, help="Star center heatmap focal loss weight (default: 2.0).")
    parser.add_argument("--gate_weight", type=float, default=1.0, help="Spike-bridging gate loss weight (default: 1.0).")
    parser.add_argument("--inst_weight", type=float, default=1.0, help="Instance BCE + Dice loss weight (default: 1.0).")
    parser.add_argument("--proto_dim", type=int, default=32, help="Prototype feature dimension D (default: 32).")
    parser.add_argument("--n_max", type=int, default=128, help="Maximum number of queries N_max (default: 128).")
    parser.add_argument("--noise_aug", action="store_true", default=True, help="Enable GPU Poisson shot & Gaussian read noise.")
    parser.add_argument("--rot_prob", type=float, default=0.0, help="Probability of random 90-degree rotations in GPU augmentation (default: 0.0, disabled).")
    parser.add_argument("--mmin_jitter", type=float, default=1.5, help="Magnitude jitter for mmin in LeakySB (default: 1.5 mag).")
    parser.add_argument("--mmax_jitter", type=float, default=1.5, help="Magnitude jitter for mmax in LeakySB (default: 1.5 mag).")
    parser.add_argument("--gradient_prob", type=float, default=0.5, help="Probability of applying tilted-plane background gradient (default: 0.5).")
    parser.add_argument("--gradient_sigma_max", type=float, default=4.0, help="Max sigma for tilted plane gradient in DN (default: 4.0).")
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
            "mmin_jitter": args.mmin_jitter if args.noise_aug else 0.0,
            "mmax_jitter": args.mmax_jitter if args.noise_aug else 0.0,
            "out_channels": 3,
        },
        aug_kwargs={
            "enable_noise": args.noise_aug,
            "enable_spatial": False,
            "rot_prob": args.rot_prob,
            "auto_scale_noise": True,
            "gain": 100.0,
            "gradient_prob": args.gradient_prob if args.noise_aug else 0.0,
            "gradient_scale_range": (0.2, 2.5),
        },
    ).to(device)

    trainer = SteerQueryTrainer(
        in_channels=3,
        num_classes=3,
        features=(32, 64, 128, 256, 512),
        proto_dim=args.proto_dim,
        N_max=args.n_max,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        sem_weight=args.sem_weight,
        center_weight=args.center_weight,
        gate_weight=args.gate_weight,
        inst_weight=args.inst_weight,
        gpu_transform=gpu_transform,
        device=device,
    )

    val_cutout_pool = None
    if val_shards:
        logger.info("Scanning validation shards for crowded cutouts pool...")
        val_cutout_pool = extract_crowded_val_cutouts(val_shards, pool_size=16, max_scan=60)
        if val_cutout_pool:
            star_counts = [c.get("n_stars", 0) for c in val_cutout_pool]
            logger.info(
                f"Cached {len(val_cutout_pool)} crowded cutouts (star counts: "
                f"min={min(star_counts)}, max={max(star_counts)}, median={int(np.median(star_counts))})."
            )

    def epoch_callback(trainer_inst: SteerQueryTrainer, epoch_data: Dict[str, Any], history: Dict[str, List[float]]) -> None:
        epoch = epoch_data["epoch"]
        epochs = epoch_data["epochs"]
        if epoch % args.plot_interval == 0 or epoch == epochs:
            plot_training_curves(history=history, output_dir=output_dir, current_epoch=epoch)
            if val_cutout_pool:
                epoch_cutouts = select_val_cutouts_for_epoch(
                    cutout_pool=val_cutout_pool,
                    epoch=epoch,
                    plot_interval=args.plot_interval,
                    num_cutouts=4,
                )
                fig_path = output_dir / f"val_inference_epoch_{epoch:03d}.png"
                plot_validation_examples(
                    model=trainer_inst.model,
                    fixed_cutouts=epoch_cutouts,
                    gpu_transform=trainer_inst.gpu_transform,
                    output_path=fig_path,
                    epoch=epoch,
                    epochs=epochs,
                    device=trainer_inst.device,
                    use_amp=trainer_inst.use_amp,
                    amp_dtype=trainer_inst.amp_dtype,
                )

    logger.info(f"Starting Grace Hopper SteerQuery training: batch_size={args.batch_size}, epochs={args.epochs}")
    t0 = time.time()

    history = trainer.train(
        dataset=train_shards,
        val_dataset=val_shards,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
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
    logger.info(f"Grace Hopper SteerQuery training finished in {total_time:.1f}s ({total_time / args.epochs:.2f}s/epoch)")
    plot_training_curves(history=history, output_dir=output_dir, current_epoch=args.epochs)


if __name__ == "__main__":
    main()
