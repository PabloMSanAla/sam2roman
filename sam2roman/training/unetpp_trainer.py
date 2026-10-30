"""
Training engine for UNet++ (Nested U-Net) on astronomical FITS datasets.
Supports 3-class boundary-aware semantic segmentation (Background, Interior, Boundary).
Optimized automatically for Apple Silicon MPS and NVIDIA CUDA GPUs (A100 / L4 / T4).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sam2roman.models.unetpp import UNetPlusPlus
from sam2roman.training.losses import MultiClassCombinedLoss
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.gpu_transforms import GPUAstroTransform
from sam2roman.training.webdataset_loader import create_webdataset_loader
from sam2roman.training.metrics import evaluate_batch_metrics

logger = logging.getLogger(__name__)


def _resolve_dataset_size_string(dataset: Any, is_train: bool = True) -> str:
    """Format human-readable dataset sample count (resolves WebDataset manifest when available)."""
    if dataset is None:
        return "None"
    if not isinstance(dataset, (str, Path)) and hasattr(dataset, "__len__"):
        return f"{len(dataset):,}"

    if isinstance(dataset, (str, Path)):
        ds_str = str(dataset)
        base_path = Path(ds_str.split("*")[0]).resolve()
        for candidate_dir in [base_path, base_path.parent, base_path.parent.parent]:
            manifest_file = candidate_dir / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        meta = json.load(f)
                    key = "train_cutouts" if is_train else "val_cutouts"
                    if key in meta:
                        return f"{meta[key]:,} (streaming WebDataset via manifest)"
                    elif "total_cutouts" in meta:
                        return f"{meta['total_cutouts']:,} (streaming WebDataset)"
                except Exception:
                    pass
        return "Streaming WebDataset"

    return "Streaming DataLoader"


class UNetPPTrainer:
    """
    Trainer for boundary-aware UNet++ on astronomical datasets.

    Automatically detects accelerator hardware:
    - **NVIDIA CUDA (A100 / L4 / T4)**: Enables TensorFloat-32 (TF32), cuDNN benchmark,
      native bfloat16 Automatic Mixed Precision (AMP), and pinned DataLoader memory.
    - **Apple Silicon MPS**: Executes on Metal Performance Shaders in full FP32.
    - **CPU**: Standard float32 fallback.

    Parameters
    ----------
    model : UNetPlusPlus, optional
        Pre-configured UNetPlusPlus instance. If None, initialized with in_channels=3, num_classes=3.
    in_channels : int, default 3
        Number of input channels (3 for leakySB RGB, 1 for raw FITS float32).
    num_classes : int, default 3
        Number of classes (0: Background, 1: Source interior, 2: Boundary/edge).
    features : tuple of int, default (32, 64, 128, 256, 512)
        Number of channels per level.
    deep_supervision : bool, default False
        Whether to train with deep supervision across intermediate nodes.
    learning_rate : float, default 1e-4
        Initial AdamW learning rate (lowered to 1e-4 to prevent validation bounce).
    weight_decay : float, default 1e-2
        AdamW weight decay regularizing UNet++ nested dense skip coefficients.
    grad_clip_norm : float, default 1.0
        Maximum norm for gradient clipping.
    class_weights : tuple or tensor, optional
        Class weights for loss balancing: [Background, Interior, Boundary].
        Default: (0.5, 2.5, 8.0) down-weighting sky and prioritizing fine boundaries.
    boundary_tolerance : int, default 1
        Pixel distance tolerance for relaxed boundary IoU validation metric (default: 1 px).
    gpu_transform : nn.Module or bool, default True
        GPU-accelerated transformations (leakySB, noise, augmentations).
    device : str or torch.device, optional
        Target device ('cuda', 'mps', 'cpu'). Auto-detected if None.
    """

    def __init__(
        self,
        model: Optional[UNetPlusPlus] = None,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        deep_supervision: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        grad_clip_norm: float = 1.0,
        class_weights: Optional[Union[Tuple[float, ...], List[float], torch.Tensor]] = None,
        boundary_tolerance: int = 1,
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

        # Optimize for CUDA architectures (A100, Hopper, Ada Lovelace)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self.use_amp = True
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            # Mixed precision safeguard: GradScaler for float16 (bfloat16 does not require scaling)
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
            self.model = UNetPlusPlus(
                in_channels=in_channels,
                num_classes=num_classes,
                features=features,
                deep_supervision=deep_supervision,
            ).to(self.device)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.features = features
        self.grad_clip_norm = grad_clip_norm
        self.boundary_tolerance = boundary_tolerance
        self.last_eval_metrics: Dict[str, float] = {}

        # Setup loss weights (prioritize boundary over sky background)
        if class_weights is None:
            weights_tensor = torch.tensor([0.5, 2.5, 8.0], dtype=torch.float32, device=self.device)
        else:
            weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=self.device)

        self.criterion = MultiClassCombinedLoss(
            class_weights=weights_tensor,
            ce_weight=1.0,
            dice_weight=1.0,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler: Optional[Any] = None
        self.learning_rate = learning_rate

        # GPU-Accelerated Astronomical Transforms (leakySB + noise/spatial augmentations)
        if isinstance(gpu_transform, nn.Module):
            self.gpu_transform = gpu_transform.to(self.device)
        elif gpu_transform is True:
            self.gpu_transform = GPUAstroTransform().to(self.device)
        else:
            self.gpu_transform = None

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batches for UNet++ (stacks image and target tensors)."""
        images = torch.stack([b["image"] for b in batch])
        targets = torch.stack([b["target"] for b in batch])
        res: Dict[str, Any] = {
            "image": images,
            "target": targets,
        }
        if "raw_data" in batch[0]:
            res["raw_data"] = [b["raw_data"] for b in batch]
        if "raw_mask" in batch[0]:
            res["raw_mask"] = [b["raw_mask"] for b in batch]
        return res

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch with mixed precision, gradient clipping, and NaN safeguards."""
        self.model.train()
        if self.gpu_transform is not None:
            self.gpu_transform.train()
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            if "image" in batch:
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
            elif "raw" in batch:
                raw = batch["raw"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                if self.gpu_transform is not None:
                    images, targets = self.gpu_transform(raw, targets)
                else:
                    images = raw
            else:
                raise KeyError(f"Expected 'image' or 'raw' key in batch, got {list(batch.keys())}")

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(images)
                    if self.deep_supervision and isinstance(outputs, list):
                        loss = sum(self.criterion(out, targets) for out in outputs) / len(outputs)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                if self.deep_supervision and isinstance(outputs, list):
                    loss = sum(self.criterion(out, targets) for out in outputs) / len(outputs)
                else:
                    loss = self.criterion(outputs, targets)

            # Mixed precision & numerical stability safeguard: check for NaNs/Infs
            if not torch.isfinite(loss):
                logger.warning(
                    f"[sam2roman] Non-finite loss detected ({loss.item()}) at step {steps}. Skipping backward step."
                )
                self.optimizer.zero_grad()
                continue

            # Backward pass with GradScaler for float16 mixed precision
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm
                    )
                self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(1, steps)

    def evaluate(
        self,
        dataloader: DataLoader,
        boundary_tolerance: Optional[int] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate on validation dataset using distance-tolerant and soft boundary metrics.

        Parameters
        ----------
        dataloader : DataLoader
            Validation dataloader.
        boundary_tolerance : int, optional
            Pixel slack tolerance (defaults to self.boundary_tolerance, e.g. 1 or 2 px).

        Returns
        -------
        Tuple[float, float, float, float]:
            (val_loss, macro_iou, interior_iou, relaxed_boundary_iou)
        """
        self.model.eval()
        if self.gpu_transform is not None:
            self.gpu_transform.eval()
        total_loss = 0.0
        steps = 0

        tol = self.boundary_tolerance if boundary_tolerance is None else boundary_tolerance

        total_interior_iou = 0.0
        total_relaxed_boundary_iou = 0.0
        total_strict_boundary_iou = 0.0
        total_soft_boundary_iou = 0.0
        total_macro_iou = 0.0

        with torch.no_grad():
            for batch in dataloader:
                if "image" in batch:
                    images = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["target"].to(self.device, non_blocking=True)
                elif "raw" in batch:
                    raw = batch["raw"].to(self.device, non_blocking=True)
                    targets = batch["target"].to(self.device, non_blocking=True)
                    if self.gpu_transform is not None:
                        images, targets = self.gpu_transform(raw, targets)
                    else:
                        images = raw
                else:
                    raise KeyError(f"Expected 'image' or 'raw' key in batch, got {list(batch.keys())}")

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        logits = self.model(images)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()

                metrics = evaluate_batch_metrics(logits, targets, boundary_tolerance=tol)
                total_interior_iou += metrics["interior_iou"]
                total_relaxed_boundary_iou += metrics["relaxed_boundary_iou"]
                total_strict_boundary_iou += metrics["strict_boundary_iou"]
                total_soft_boundary_iou += metrics["soft_boundary_iou"]
                total_macro_iou += metrics["macro_iou"]
                steps += 1

        n = max(1, steps)
        self.last_eval_metrics = {
            "val_loss": total_loss / n,
            "macro_iou": total_macro_iou / n,
            "interior_iou": total_interior_iou / n,
            "relaxed_boundary_iou": total_relaxed_boundary_iou / n,
            "strict_boundary_iou": total_strict_boundary_iou / n,
            "soft_boundary_iou": total_soft_boundary_iou / n,
        }

        return (
            total_loss / n,
            total_macro_iou / n,
            total_interior_iou / n,
            total_relaxed_boundary_iou / n,
        )

    def train(
        self,
        dataset: Union[AstronomicalDataset, DataLoader, str, Path],
        epochs: int = 10,
        batch_size: int = 8,
        val_dataset: Optional[Union[AstronomicalDataset, DataLoader, str, Path]] = None,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 2,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full training routine for UNet++.

        Parameters
        ----------
        dataset : AstronomicalDataset, DataLoader, str, or Path
            Training dataset: can be an AstronomicalDataset, an active DataLoader (e.g. WebDataset),
            or a path/glob string pointing to WebDataset (.tar) shards.
        epochs : int, default 10
            Number of training epochs.
        batch_size : int, default 8
            Mini-batch size (can be set to 16–64 on Grace Hopper H100 or A100).
        val_dataset : AstronomicalDataset, DataLoader, str, or Path, optional
            Optional validation dataset, loader, or WebDataset shard path.
        scheduler_type : str, default 'cosine'
            Learning rate scheduler: 'cosine' (with linear warmup), 'plateau' (ReduceLROnPlateau), or 'none'.
        warmup_epochs : int, default 2
            Number of warmup epochs for cosine schedule to prevent validation bounce.
        output_dir : str or Path, optional
            Directory to save checkpoints.
        verbose : bool, default True
            Print metrics and progress per epoch.
        num_workers : int, default 0
            Number of dataloader workers.
        pin_memory : bool, optional
            Pin host memory for DMA transfers (defaults to True on CUDA).
        shuffle : bool, default True
            Whether to shuffle training crops every epoch.
        """
        if pin_memory is None:
            pin_memory = (self.device.type == "cuda")

        # Setup learning rate schedule
        if scheduler_type == "cosine":
            if warmup_epochs > 0 and epochs > warmup_epochs:
                sched_warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.2,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs - warmup_epochs,
                    eta_min=self.learning_rate * 0.05,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[sched_warmup, sched_cosine],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs,
                    eta_min=self.learning_rate * 0.05,
                )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=self.learning_rate * 0.01,
            )
        else:
            self.scheduler = None

        # Build or wrap train DataLoader
        if isinstance(dataset, DataLoader):
            train_loader = dataset
        elif isinstance(dataset, (str, Path)):
            train_loader = create_webdataset_loader(
                urls=dataset,
                batch_size=batch_size,
                is_train=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self.collate_fn,
            )

        # Build or wrap validation DataLoader
        if val_dataset is None:
            val_loader = None
        elif isinstance(val_dataset, DataLoader):
            val_loader = val_dataset
        elif isinstance(val_dataset, (str, Path)):
            val_loader = create_webdataset_loader(
                urls=val_dataset,
                batch_size=batch_size,
                is_train=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self.collate_fn,
            )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_iou": [],
            "val_interior_iou": [],
            "val_boundary_iou": [],
            "val_strict_boundary_iou": [],
            "val_soft_boundary_iou": [],
            "learning_rate": [],
        }

        best_val_loss = float("inf")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        n_train_str = _resolve_dataset_size_string(dataset, is_train=True)
        n_val_str = _resolve_dataset_size_string(val_dataset, is_train=False)

        if verbose:
            print(f"[sam2roman] Starting UNet++ training for {epochs} epochs on {self.device}...")
            print(f"  -> Model Architecture:  UNet++ (Nested U-Net, 3 classes: bg, interior, boundary)")
            print(f"  -> Trainable weights:   {trainable_params:,} parameters")
            print(f"  -> Training cutouts:    {n_train_str}")
            if val_dataset is not None:
                print(f"  -> Validation cutouts:  {n_val_str}")
            print(f"  -> Batch size:          {batch_size}")
            print(f"  -> Initial LR:          {self.learning_rate:.2e} (AdamW weight_decay={self.optimizer.param_groups[0]['weight_decay']:.2e})")
            print(f"  -> LR Schedule:         {scheduler_type} (warmup_epochs={warmup_epochs})")
            print(f"  -> Grad Clipping:       max_norm={self.grad_clip_norm}")
            print(f"  -> Boundary Metric:     Distance-Tolerant IoU (tolerance={self.boundary_tolerance}px) + Soft Continuous IoU")
            print(f"  -> DataLoader:          shuffle={shuffle}, num_workers={num_workers}, pin_memory={pin_memory}")
            if self.device.type == "cuda":
                print(f"  -> CUDA Acceleration:   TF32=True, AMP ({self.amp_dtype}), GradScaler={self.scaler.is_enabled()}")
            elif self.device.type == "mps":
                print(f"  -> Apple Silicon MPS:   Metal Performance Shaders active (FP32)")
            print("-" * 75)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["learning_rate"].append(current_lr)

            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            elapsed = time.time() - t0

            val_str = ""
            val_loss = None
            if val_loader is not None:
                val_loss, macro_iou, int_iou, bound_iou = self.evaluate(val_loader)
                m = self.last_eval_metrics
                history["val_loss"].append(val_loss)
                history["val_macro_iou"].append(macro_iou)
                history["val_interior_iou"].append(int_iou)
                history["val_boundary_iou"].append(bound_iou)
                history["val_strict_boundary_iou"].append(m["strict_boundary_iou"])
                history["val_soft_boundary_iou"].append(m["soft_boundary_iou"])

                val_str = (
                    f" | Val Loss: {val_loss:.4f} | Macro IoU: {macro_iou:.4f} "
                    f"(Interior: {int_iou:.3f}, Edge [tol={self.boundary_tolerance}px]: {bound_iou:.3f}, "
                    f"Strict: {m['strict_boundary_iou']:.3f}, Soft: {m['soft_boundary_iou']:.3f})"
                )

                if output_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = Path(output_dir) / "unetpp_best.pt"
                    self.save_checkpoint(best_path)

            # Step learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] - LR: {current_lr:.2e} - "
                    f"Train Loss: {train_loss:.4f}{val_str} ({elapsed:.1f}s)"
                )

            if output_dir:
                save_path = Path(output_dir) / f"unetpp_epoch_{epoch}.pt"
                self.save_checkpoint(save_path)

        if verbose:
            print("-" * 75)
            print("[sam2roman] UNet++ training completed successfully!")

        return history

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save UNet++ checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "features": self.features,
                "deep_supervision": self.deep_supervision,
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load UNet++ checkpoint weights."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        logger.info(f"Loaded UNet++ checkpoint from: {path}")
