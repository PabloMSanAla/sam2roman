"""
Trainer engine for Multi-Task Center Offset UNet++ on astronomical datasets.
=============================================================================

Optimized for NVIDIA Grace Hopper (GH200):
- TF32 & cuDNN benchmark
- Native bfloat16 Automatic Mixed Precision (AMP)
- Asynchronous CPU worker pre-computation ring buffer (72 ARM cores, 480GB RAM)
- Fast non-blocking DMA streaming over NVLink-C2C
- In-VRAM GPUAstroTransform (<0.05 ms per batch)
- Periodic validation tracking: Center Peak F1, Offset EPE, Foreground IoU
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sam2roman.models.unetpp_offset import UNetPlusPlusCenterOffset
from sam2roman.training.offset_losses import CenterOffsetCompositeLoss
from sam2roman.training.offset_target import generate_center_offset_batch
from sam2roman.training.offset_metrics import (
    extract_center_peaks,
    evaluate_center_detection,
    compute_offset_epe,
)
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import create_webdataset_loader

logger = logging.getLogger("sam2roman.center_offset_trainer")


def _is_webdataset_shards(ds: Any) -> bool:
    """Check if dataset argument represents WebDataset shard paths or patterns."""
    if isinstance(ds, (str, Path)):
        return True
    if isinstance(ds, (list, tuple)) and len(ds) > 0 and isinstance(ds[0], (str, Path)):
        return True
    return False


class CenterOffsetTrainer:
    """
    Trainer for Center Offset Voting instance segmentation UNet++.

    Parameters
    ----------
    model : UNetPlusPlusCenterOffset, optional
        Multi-task model. Auto-instantiated if None.
    in_channels : int, default 3
        Input channels (3 for leakySB RGB).
    semantic_classes : int, default 1
        Foreground classes (1 for binary BCE/Dice).
    features : tuple of int, default (32, 64, 128, 256, 512)
        Feature channels per level.
    use_cca : bool, default True
        Whether to enable Criss-Cross Attention at bottleneck and deep decoders.
    learning_rate : float, default 6e-5
        Initial AdamW learning rate.
    weight_decay : float, default 1e-2
        AdamW weight decay.
    grad_clip_norm : float, default 1.0
        Gradient clipping threshold.
    semantic_weight : float, default 1.0
        Weight for semantic foreground loss.
    center_weight : float, default 2.0
        Weight for center heatmap focal loss.
    offset_weight : float, default 1.0
        Weight for offset masked Smooth L1 loss.
    angle_weight : float, default 0.5
        Weight for directional angular cosine loss (diffraction spikes).
    center_sigma : float, default 2.0
        Gaussian sigma for target center peaks.
    gpu_transform : nn.Module or bool, default True
        GPU-accelerated leakySB stretch and astronomical noise transforms.
    device : str or torch.device, optional
        Target device ('cuda', 'mps', 'cpu'). Auto-detected if None.
    """

    def __init__(
        self,
        model: Optional[UNetPlusPlusCenterOffset] = None,
        in_channels: int = 3,
        semantic_classes: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        use_cca: bool = True,
        learning_rate: float = 6e-5,
        weight_decay: float = 1e-2,
        grad_clip_norm: float = 1.0,
        semantic_weight: float = 1.0,
        center_weight: float = 2.0,
        offset_weight: float = 1.0,
        angle_weight: float = 0.5,
        center_sigma: float = 2.0,
        gpu_transform: Optional[Union[nn.Module, bool]] = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Optimize for CUDA Hopper / Ada architectures
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self.use_amp = True
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16)
            )
        else:
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)

        # Initialize model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = UNetPlusPlusCenterOffset(
                in_channels=in_channels,
                semantic_classes=semantic_classes,
                features=features,
                use_cca=use_cca,
            ).to(self.device)

        self.in_channels = in_channels
        self.semantic_classes = semantic_classes
        self.features = features
        self.use_cca = use_cca
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.center_sigma = center_sigma

        # Multi-task loss criterion
        self.criterion = CenterOffsetCompositeLoss(
            semantic_weight=semantic_weight,
            center_weight=center_weight,
            offset_weight=offset_weight,
            angle_weight=angle_weight,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # In-VRAM GPU transform
        if isinstance(gpu_transform, nn.Module):
            self.gpu_transform = gpu_transform.to(self.device)
        elif gpu_transform is True:
            self.gpu_transform = GPUAstroTransform().to(self.device)
        else:
            self.gpu_transform = None

    def _prepare_targets(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract or generate (fg_target, center_target, offset_target) from batch.
        """
        # If precomputed in shard:
        if "fg" in batch and "center" in batch and "offset" in batch:
            fg_t = batch["fg"].to(self.device, non_blocking=True).float()
            c_t = batch["center"].to(self.device, non_blocking=True).float()
            o_t = batch["offset"].to(self.device, non_blocking=True).float()
            return fg_t, c_t, o_t

        # Otherwise, dynamically generate from integer instance masks (Option A)
        target_mask = batch["target"]  # (B, H, W)
        fg_t, c_t, o_t = generate_center_offset_batch(
            target_mask,
            sigma=self.center_sigma,
            device=self.device,
        )
        return fg_t, c_t, o_t

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        if self.gpu_transform is not None:
            self.gpu_transform.train()

        total_loss = 0.0
        total_sem = 0.0
        total_center = 0.0
        total_offset = 0.0
        total_ang = 0.0
        steps = 0

        for batch in dataloader:
            if "image" in batch:
                images = batch["image"].to(self.device, non_blocking=True)
            elif "raw" in batch:
                raw = batch["raw"].to(self.device, non_blocking=True)
                if self.gpu_transform is not None:
                    # Stretches raw counts into normalized [0, 1] RGB in VRAM (<0.05 ms)
                    images, _ = self.gpu_transform(raw, None)
                else:
                    images = raw
            else:
                raise KeyError(f"Expected 'image' or 'raw' key in batch, got {list(batch.keys())}")

            fg_t, c_t, o_t = self._prepare_targets(batch)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    preds = self.model(images)
                    loss_dict = self.criterion(preds, fg_t, c_t, o_t)
                    loss = loss_dict["loss"]
            else:
                preds = self.model(images)
                loss_dict = self.criterion(preds, fg_t, c_t, o_t)
                loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss ({loss.item()}) at step {steps}. Skipping backward.")
                self.optimizer.zero_grad()
                continue

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()
            total_sem += loss_dict["sem_loss"].item()
            total_center += loss_dict["center_loss"].item()
            total_offset += loss_dict["offset_loss"].item()
            total_ang += loss_dict["angle_loss"].item()
            steps += 1

        n = max(1, steps)
        return {
            "loss": total_loss / n,
            "sem_loss": total_sem / n,
            "center_loss": total_center / n,
            "offset_loss": total_offset / n,
            "angle_loss": total_ang / n,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate validation metrics."""
        self.model.eval()
        if self.gpu_transform is not None:
            self.gpu_transform.eval()

        total_loss = 0.0
        total_f1 = 0.0
        total_mae = 0.0
        total_epe = 0.0
        total_fg_iou = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                if "image" in batch:
                    images = batch["image"].to(self.device, non_blocking=True)
                elif "raw" in batch:
                    raw = batch["raw"].to(self.device, non_blocking=True)
                    if self.gpu_transform is not None:
                        images, _ = self.gpu_transform(raw, None)
                    else:
                        images = raw
                else:
                    raise KeyError(f"Expected 'image' or 'raw' in batch, got {list(batch.keys())}")

                fg_t, c_t, o_t = self._prepare_targets(batch)

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        preds = self.model(images)
                        loss_dict = self.criterion(preds, fg_t, c_t, o_t)
                else:
                    preds = self.model(images)
                    loss_dict = self.criterion(preds, fg_t, c_t, o_t)

                total_loss += loss_dict["loss"].item()

                # Compute foreground IoU
                pred_fg_mask = (torch.sigmoid(preds["semantic"]) >= 0.5).squeeze(1)
                gt_fg = (fg_t > 0).squeeze(1)
                intersection = (pred_fg_mask & gt_fg).sum().float().item()
                union = (pred_fg_mask | gt_fg).sum().float().item()
                iou = intersection / max(1.0, union)
                total_fg_iou += iou

                # Compute Offset EPE
                epe = compute_offset_epe(preds["offset"], o_t, fg_t)
                total_epe += epe

                # Compute Center Peak F1 on first item of batch for speed
                pred_c_prob = torch.sigmoid(preds["center"][0, 0]).float().cpu().numpy()
                peaks = extract_center_peaks(pred_c_prob, threshold=0.3, min_distance=3)

                # Extract ground truth centroids from target mask
                gt_mask_0 = batch["target"][0].detach().cpu().numpy()
                gt_ids = np.unique(gt_mask_0)
                gt_ids = gt_ids[gt_ids > 0]
                gt_centroids = []
                for gid in gt_ids:
                    ys, xs = np.where(gt_mask_0 == gid)
                    if len(ys) > 0:
                        gt_centroids.append((float(np.mean(ys)), float(np.mean(xs))))

                det_res = evaluate_center_detection(peaks, gt_centroids, tolerance=2.0)
                total_f1 += det_res["f1"]
                total_mae += det_res["mae"]
                steps += 1

        n = max(1, steps)
        return {
            "val_loss": total_loss / n,
            "fg_iou": total_fg_iou / n,
            "center_f1": total_f1 / n,
            "center_mae": total_mae / n,
            "offset_epe": total_epe / n,
        }

    def train(
        self,
        dataset: Any,
        val_dataset: Optional[Any] = None,
        epochs: int = 50,
        batch_size: int = 32,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 3,
        num_workers: int = 16,
        prefetch_factor: int = 8,
        output_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[str] = "epoch_log.txt",
        epoch_callback: Optional[Callable[[Any, Dict[str, Any], Dict[str, List[float]]], None]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Execute full multi-epoch training run.
        """
        # Build DataLoaders
        if _is_webdataset_shards(dataset):
            train_loader = create_webdataset_loader(
                urls=dataset,
                batch_size=batch_size,
                is_train=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True if num_workers > 0 else False,
            )
        elif isinstance(dataset, DataLoader):
            train_loader = dataset
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if self.device.type == "cuda" else False,
            )

        val_loader = None
        if val_dataset is not None:
            if _is_webdataset_shards(val_dataset):
                val_loader = create_webdataset_loader(
                    urls=val_dataset,
                    batch_size=batch_size,
                    is_train=False,
                    num_workers=max(1, num_workers // 2),
                    prefetch_factor=prefetch_factor,
                    persistent_workers=True if num_workers > 0 else False,
                )
            elif isinstance(val_dataset, DataLoader):
                val_loader = val_dataset
            else:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=max(1, num_workers // 2),
                    pin_memory=True if self.device.type == "cuda" else False,
                )

        # Scheduler
        if scheduler_type == "cosine":
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-7
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = None

        out_path = Path(output_dir) if output_dir else Path("checkpoints")
        out_path.mkdir(parents=True, exist_ok=True)
        log_fp = (out_path / log_file) if log_file else None

        history: Dict[str, List[float]] = {
            "loss": [],
            "sem_loss": [],
            "center_loss": [],
            "offset_loss": [],
            "angle_loss": [],
            "val_loss": [],
            "fg_iou": [],
            "center_f1": [],
            "offset_epe": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        logger.info(f"Beginning Center Offset UNet++ training: {epochs} epochs, batch_size={batch_size}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)

            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            history["loss"].append(train_metrics["loss"])
            history["sem_loss"].append(train_metrics["sem_loss"])
            history["center_loss"].append(train_metrics["center_loss"])
            history["offset_loss"].append(train_metrics["offset_loss"])
            history["angle_loss"].append(train_metrics["angle_loss"])
            history["lr"].append(current_lr)

            if val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])
                history["fg_iou"].append(val_metrics["fg_iou"])
                history["center_f1"].append(val_metrics["center_f1"])
                history["offset_epe"].append(val_metrics["offset_epe"])
            else:
                history["val_loss"].append(train_metrics["loss"])
                history["fg_iou"].append(0.0)
                history["center_f1"].append(0.0)
                history["offset_epe"].append(0.0)

            dt = time.time() - t0
            val_loss = val_metrics.get("val_loss", train_metrics["loss"])
            f1 = val_metrics.get("center_f1", 0.0)
            epe = val_metrics.get("offset_epe", 0.0)

            msg = (
                f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_metrics['loss']:.4f} "
                f"(sem: {train_metrics['sem_loss']:.3f}, ctr: {train_metrics['center_loss']:.3f}, "
                f"off: {train_metrics['offset_loss']:.3f}, ang: {train_metrics['angle_loss']:.3f}) | "
                f"Val Loss: {val_loss:.4f} | Center F1: {f1:.3f} | Offset EPE: {epe:.2f}px | "
                f"LR: {current_lr:.2e} | {dt:.1f}s"
            )
            if verbose:
                logger.info(msg)
            if log_fp:
                with open(log_fp, "a") as f:
                    f.write(msg + "\n")

            # Save latest checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "in_channels": self.in_channels,
                    "semantic_classes": self.semantic_classes,
                    "features": self.features,
                    "use_cca": self.use_cca,
                },
                out_path / "center_offset_latest.pt",
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": val_loss,
                        "in_channels": self.in_channels,
                        "semantic_classes": self.semantic_classes,
                        "features": self.features,
                        "use_cca": self.use_cca,
                    },
                    out_path / "center_offset_best.pt",
                )

            if epoch_callback is not None:
                epoch_data = {"epoch": epoch, "epochs": epochs, "val_metrics": val_metrics}
                epoch_callback(self, epoch_data, history)

        with open(out_path / "training_metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        return history
