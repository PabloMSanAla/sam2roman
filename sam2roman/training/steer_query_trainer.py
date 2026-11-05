"""
Trainer engine for SteerQuery Star-Deblending UNet++.
=====================================================

Optimized for NVIDIA Grace Hopper (GH200):
- TF32 & native bfloat16 Automatic Mixed Precision (AMP)
- Two-speed optimizer (0.1x LR on geometric angle and curvature steering parameters)
- Teacher-forcing query warmup during early epochs
- Direct GPU integer instance mask assembly without CPU voting/clustering
- Continuous validation tracking: Foreground IoU, Star Centroid F1, Gate Accuracy
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sam2roman.models.steer_query_head import UNetPlusPlusSteerQuery
from sam2roman.losses.steer_query_loss import (
    SteerQueryLoss,
    build_steer_query_optimizer,
    render_star_center_heatmap,
)
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import create_webdataset_loader

logger = logging.getLogger("sam2roman.steer_query_trainer")


def _is_webdataset_shards(ds: Any) -> bool:
    """Check if dataset argument represents WebDataset shard paths or patterns."""
    if isinstance(ds, (str, Path)):
        return True
    if isinstance(ds, (list, tuple)) and len(ds) > 0 and isinstance(ds[0], (str, Path)):
        return True
    return False


def derive_semantic_3class_from_instances(
    inst_gt: torch.Tensor,
    edge_width: int = 2,
) -> torch.Tensor:
    """
    Derive 3-class semantic masks from integer instance masks:
    0: Sky/background and unmasked galaxies
    1: Star core/interior
    2: Star boundary/diffraction spikes
    """
    b, h, w = inst_gt.shape
    device = inst_gt.device
    sem_gt = torch.zeros((b, h, w), device=device, dtype=torch.long)

    # Foreground pixels
    fg_mask = (inst_gt > 0).float().unsqueeze(1)  # (B, 1, H, W)
    if not fg_mask.any():
        return sem_gt

    # Morphological erosion to separate interior from edges
    pad = edge_width
    pooled = -F.max_pool2d(-fg_mask, kernel_size=2 * pad + 1, stride=1, padding=pad)
    eroded = (pooled > 0.5).squeeze(1)

    fg_bool = fg_mask.squeeze(1).bool()
    sem_gt[fg_bool] = 2  # Edge / spike
    sem_gt[eroded] = 1   # Interior

    return sem_gt


class SteerQueryTrainer:
    """
    Trainer for SteerQuery UNet++ star deblending and instance segmentation.
    """

    def __init__(
        self,
        model: Optional[UNetPlusPlusSteerQuery] = None,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        proto_dim: int = 32,
        N_max: int = 128,
        learning_rate: float = 6e-5,
        weight_decay: float = 1e-2,
        grad_clip_norm: float = 1.0,
        sem_weight: float = 1.0,
        center_weight: float = 2.0,
        gate_weight: float = 1.0,
        inst_weight: float = 1.0,
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

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = UNetPlusPlusSteerQuery(
                in_channels=in_channels,
                num_classes=num_classes,
                features=features,
                proto_dim=proto_dim,
                N_max=N_max,
            ).to(self.device)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.proto_dim = proto_dim
        self.N_max = N_max
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm

        self.criterion = SteerQueryLoss(
            sem_weight=sem_weight,
            center_weight=center_weight,
            gate_weight=gate_weight,
            inst_weight=inst_weight,
        ).to(self.device)

        self.optimizer = build_steer_query_optimizer(
            self.model, base_lr=learning_rate, weight_decay=weight_decay
        )

        if isinstance(gpu_transform, nn.Module):
            self.gpu_transform = gpu_transform.to(self.device)
        elif gpu_transform is True:
            self.gpu_transform = GPUAstroTransform().to(self.device)
        else:
            self.gpu_transform = None

    def _prepare_inputs_and_targets(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[Tuple[float, float]]]]:
        """Extract and prepare images, instance ground truth, semantics, and centroids."""
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

        inst_gt = batch["target"].to(self.device, non_blocking=True).int()

        if "semantic" in batch:
            sem_gt = batch["semantic"].to(self.device, non_blocking=True).long()
        else:
            sem_gt = derive_semantic_3class_from_instances(inst_gt)

        H_star_gt, batch_centroids = render_star_center_heatmap(inst_gt, sigma=2.0)
        return images, inst_gt, sem_gt, H_star_gt, batch_centroids

    def train_epoch(
        self,
        dataloader: DataLoader,
        grad_accum_steps: int = 1,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        if self.gpu_transform is not None:
            self.gpu_transform.train()

        total_loss = 0.0
        total_sem = 0.0
        total_ctr = 0.0
        total_gate = 0.0
        total_inst = 0.0
        steps = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            images, inst_gt, sem_gt, H_star_gt, batch_centroids = self._prepare_inputs_and_targets(batch)
            centroids_tf = batch_centroids if use_teacher_forcing else None

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    preds = self.model(images, gt_centroids=centroids_tf)
                    loss_dict = self.criterion(
                        preds, inst_gt=inst_gt, sem_gt=sem_gt, H_star_gt=H_star_gt, image=images
                    )
                    loss = loss_dict["loss"] / grad_accum_steps
            else:
                preds = self.model(images, gt_centroids=centroids_tf)
                loss_dict = self.criterion(
                    preds, inst_gt=inst_gt, sem_gt=sem_gt, H_star_gt=H_star_gt, image=images
                )
                loss = loss_dict["loss"] / grad_accum_steps

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss ({loss.item() * grad_accum_steps}) at step {steps}. Skipping.")
                self.optimizer.zero_grad()
                continue

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            total_sem += loss_dict["sem_loss"].item()
            total_ctr += loss_dict["center_loss"].item()
            total_gate += loss_dict["gate_loss"].item()
            total_inst += loss_dict["inst_loss"].item()
            steps += 1

        if steps > 0 and (steps % grad_accum_steps) != 0:
            if self.scaler.is_enabled():
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()

        n = max(1, steps)
        return {
            "loss": total_loss / n,
            "sem_loss": total_sem / n,
            "center_loss": total_ctr / n,
            "gate_loss": total_gate / n,
            "inst_loss": total_inst / n,
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate validation metrics."""
        self.model.eval()
        if self.gpu_transform is not None:
            self.gpu_transform.eval()

        total_loss = 0.0
        total_fg_iou = 0.0
        total_star_f1 = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                images, inst_gt, sem_gt, H_star_gt, batch_centroids = self._prepare_inputs_and_targets(batch)

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        preds = self.model(images, gt_centroids=None)
                        loss_dict = self.criterion(
                            preds, inst_gt=inst_gt, sem_gt=sem_gt, H_star_gt=H_star_gt, image=images
                        )
                else:
                    preds = self.model(images, gt_centroids=None)
                    loss_dict = self.criterion(
                        preds, inst_gt=inst_gt, sem_gt=sem_gt, H_star_gt=H_star_gt, image=images
                    )

                total_loss += loss_dict["loss"].item()

                # 1. Foreground IoU of predicted star masks vs ground truth stars
                pred_fg = preds["instance_mask"] > 0
                gt_fg = inst_gt > 0
                inter = (pred_fg & gt_fg).sum().float().item()
                union = (pred_fg | gt_fg).sum().float().item()
                iou = inter / max(1.0, union)
                total_fg_iou += iou

                # 2. Star Detection F1 on first item of batch
                pts_gt = batch_centroids[0]
                coords_pred = preds["star_coords"][0]  # (N, 2)
                scores_pred = preds["scores"][0]
                keep = scores_pred >= 0.30
                coords_pred = coords_pred[keep]

                if len(pts_gt) == 0 and len(coords_pred) == 0:
                    f1 = 1.0
                elif len(pts_gt) == 0 or len(coords_pred) == 0:
                    f1 = 0.0
                else:
                    tp = 0
                    matched_gt = set()
                    for px, py in coords_pred.tolist():
                        for g_idx, (gx, gy) in enumerate(pts_gt):
                            if g_idx in matched_gt:
                                continue
                            dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                            if dist <= 3.0:
                                tp += 1
                                matched_gt.add(g_idx)
                                break
                    fp = len(coords_pred) - tp
                    fn = len(pts_gt) - tp
                    prec = tp / max(1, tp + fp)
                    rec = tp / max(1, tp + fn)
                    f1 = 2 * prec * rec / max(1e-8, prec + rec)

                total_star_f1 += f1
                steps += 1

        n = max(1, steps)
        return {
            "val_loss": total_loss / n,
            "fg_iou": total_fg_iou / n,
            "star_f1": total_star_f1 / n,
        }

    def train(
        self,
        dataset: Any,
        val_dataset: Optional[Any] = None,
        epochs: int = 50,
        batch_size: int = 16,
        grad_accum_steps: int = 1,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        num_workers: int = 16,
        prefetch_factor: int = 4,
        output_dir: Optional[Union[str, Path]] = None,
        log_file: Optional[str] = "epoch_log.txt",
        epoch_callback: Optional[Callable[[Any, Dict[str, Any], Dict[str, List[float]]], None]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Execute multi-epoch training with teacher-forcing query warmup."""
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
            "gate_loss": [],
            "inst_loss": [],
            "val_loss": [],
            "fg_iou": [],
            "star_f1": [],
            "lr": [],
        }

        best_val_loss = float("inf")
        logger.info(f"Beginning SteerQuery UNet++ training: {epochs} epochs, batch_size={batch_size}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            # Enable teacher forcing during warmup epochs
            use_tf = epoch <= warmup_epochs
            train_metrics = self.train_epoch(
                train_loader,
                grad_accum_steps=grad_accum_steps,
                use_teacher_forcing=use_tf,
            )

            val_metrics: Dict[str, float] = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            history["loss"].append(train_metrics["loss"])
            history["sem_loss"].append(train_metrics["sem_loss"])
            history["center_loss"].append(train_metrics["center_loss"])
            history["gate_loss"].append(train_metrics["gate_loss"])
            history["inst_loss"].append(train_metrics["inst_loss"])
            history["lr"].append(current_lr)

            if val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])
                history["fg_iou"].append(val_metrics["fg_iou"])
                history["star_f1"].append(val_metrics["star_f1"])
            else:
                history["val_loss"].append(train_metrics["loss"])
                history["fg_iou"].append(0.0)
                history["star_f1"].append(0.0)

            dt = time.time() - t0
            val_loss = val_metrics.get("val_loss", train_metrics["loss"])
            iou = val_metrics.get("fg_iou", 0.0)
            f1 = val_metrics.get("star_f1", 0.0)

            msg = (
                f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_metrics['loss']:.4f} "
                f"(sem: {train_metrics['sem_loss']:.3f}, ctr: {train_metrics['center_loss']:.3f}, "
                f"gate: {train_metrics['gate_loss']:.3f}, inst: {train_metrics['inst_loss']:.3f}) | "
                f"Val Loss: {val_loss:.4f} | Fg IoU: {iou:.3f} | Star F1: {f1:.3f} | "
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
                    "num_classes": self.num_classes,
                    "features": self.features,
                    "proto_dim": self.proto_dim,
                    "N_max": self.N_max,
                },
                out_path / "steer_query_latest.pt",
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "in_channels": self.in_channels,
                        "num_classes": self.num_classes,
                        "features": self.features,
                        "proto_dim": self.proto_dim,
                        "N_max": self.N_max,
                    },
                    out_path / "steer_query_best.pt",
                )

            if epoch_callback is not None:
                epoch_data = {"epoch": epoch, "epochs": epochs, "val_metrics": val_metrics}
                epoch_callback(self, epoch_data, history)

        with open(out_path / "training_metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        return history
