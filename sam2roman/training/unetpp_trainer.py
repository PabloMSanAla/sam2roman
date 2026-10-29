"""
Training engine for UNet++ (Nested U-Net) on astronomical FITS datasets.
Supports 3-class boundary-aware semantic segmentation (Background, Interior, Boundary).
Optimized automatically for Apple Silicon MPS and NVIDIA CUDA GPUs (A100 / L4 / T4).
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


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
    learning_rate : float, default 3e-4
        Initial AdamW learning rate.
    weight_decay : float, default 1e-4
        L2 regularization weight decay.
    class_weights : tuple or tensor, optional
        Class weights for loss balancing: [Background, Interior, Boundary].
        Default: (1.0, 2.5, 4.0) to prioritize fine boundaries and source interiors.
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
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[Union[Tuple[float, ...], List[float], torch.Tensor]] = None,
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
        else:
            self.use_amp = False
            self.amp_dtype = torch.float32

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

        # Setup loss weights
        if class_weights is None:
            weights_tensor = torch.tensor([1.0, 2.5, 4.0], dtype=torch.float32, device=self.device)
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
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
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
        """Run one training epoch."""
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(1, steps)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float, float]:
        """
        Evaluate on validation dataset.
        Returns: (val_loss, macro_iou, interior_iou, boundary_iou)
        """
        self.model.eval()
        if self.gpu_transform is not None:
            self.gpu_transform.eval()
        total_loss = 0.0
        steps = 0

        total_interior_iou = 0.0
        total_boundary_iou = 0.0
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
                preds = torch.argmax(logits, dim=1)  # (B, H, W)

                # Interior IoU (Class 1)
                p_int = (preds == 1)
                t_int = (targets == 1)
                int_inter = torch.sum(p_int & t_int).item()
                int_union = torch.sum(p_int | t_int).item()
                iou_int = (int_inter + 1e-6) / (int_union + 1e-6)

                # Boundary IoU (Class 2)
                p_bound = (preds == 2)
                t_bound = (targets == 2)
                bound_inter = torch.sum(p_bound & t_bound).item()
                bound_union = torch.sum(p_bound | t_bound).item()
                iou_bound = (bound_inter + 1e-6) / (bound_union + 1e-6)

                # Macro IoU (mean of interior and boundary)
                macro_iou = 0.5 * (iou_int + iou_bound)

                total_interior_iou += iou_int
                total_boundary_iou += iou_bound
                total_macro_iou += macro_iou
                steps += 1

        n = max(1, steps)
        return (
            total_loss / n,
            total_macro_iou / n,
            total_interior_iou / n,
            total_boundary_iou / n,
        )

    def train(
        self,
        dataset: Union[AstronomicalDataset, DataLoader, str, Path],
        epochs: int = 10,
        batch_size: int = 8,
        val_dataset: Optional[Union[AstronomicalDataset, DataLoader, str, Path]] = None,
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

        # Setup Cosine Annealing learning rate schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=self.learning_rate * 0.05,
        )

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
        }

        best_val_loss = float("inf")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        n_train_str = f"{len(dataset):,}" if hasattr(dataset, "__len__") else "Streaming WebDataset"
        n_val_str = (
            f"{len(val_dataset):,}"
            if (val_dataset is not None and hasattr(val_dataset, "__len__"))
            else ("Streaming WebDataset" if val_dataset is not None else "None")
        )

        if verbose:
            print(f"[sam2roman] Starting UNet++ training for {epochs} epochs on {self.device}...")
            print(f"  -> Model Architecture:  UNet++ (Nested U-Net, 3 classes: bg, interior, boundary)")
            print(f"  -> Trainable weights:   {trainable_params:,} parameters")
            print(f"  -> Training cutouts:    {n_train_str}")
            if val_dataset is not None:
                print(f"  -> Validation cutouts:  {n_val_str}")
            print(f"  -> Batch size:          {batch_size}")
            print(f"  -> DataLoader:          shuffle={shuffle}, num_workers={num_workers}, pin_memory={pin_memory}")
            if self.device.type == "cuda":
                print(f"  -> CUDA Acceleration:   TF32=True, AMP ({self.amp_dtype})")
            elif self.device.type == "mps":
                print(f"  -> Apple Silicon MPS:   Metal Performance Shaders active (FP32)")
            print("-" * 75)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            self.scheduler.step()
            elapsed = time.time() - t0

            val_str = ""
            if val_loader is not None:
                val_loss, macro_iou, int_iou, bound_iou = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_macro_iou"].append(macro_iou)
                history["val_interior_iou"].append(int_iou)
                history["val_boundary_iou"].append(bound_iou)

                val_str = (
                    f" | Val Loss: {val_loss:.4f} | Macro IoU: {macro_iou:.4f} "
                    f"(Interior: {int_iou:.3f}, Edge: {bound_iou:.3f})"
                )

                if output_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = Path(output_dir) / "unetpp_best.pt"
                    self.save_checkpoint(best_path)

            if verbose:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}{val_str} ({elapsed:.1f}s)")

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
