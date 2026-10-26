"""
Training and fine-tuning engine for SAM 2 on astronomical datasets.
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
from tqdm import tqdm

from sam2roman.models.sam2 import SAM2ModelWrapper, DEFAULT_SAM2_MODEL
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import CombinedSegmentationLoss

logger = logging.getLogger(__name__)


class SAM2Trainer:
    """
    Fine-tuning trainer for SAM 2 models on astronomical FITS datasets.

    By default, freezes the large Hiera image encoder and tunes the prompt encoder
    and mask decoder for domain adaptation to Roman Space Telescope imaging.

    Parameters
    ----------
    model_wrapper : SAM2ModelWrapper, optional
        Pre-configured SAM 2 model wrapper.
    learning_rate : float, default 1e-4
        Learning rate for trainable parameters.
    weight_decay : float, default 1e-4
        Weight decay for AdamW.
    freeze_vision_encoder : bool, default True
        Whether to freeze the heavy vision backbone.
    prompt_type : str, default 'point'
        Type of prompt used during training: 'point' or 'box'.
    use_amp : bool, default True
        Enable Automatic Mixed Precision (AMP / bfloat16 on NVIDIA A100/H100 GPUs).
    device : str or torch.device, optional
        Target device ('cuda', 'mps', 'cpu').
    """

    def __init__(
        self,
        model_wrapper: Optional[SAM2ModelWrapper] = None,
        model_id: str = DEFAULT_SAM2_MODEL,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_vision_encoder: bool = True,
        prompt_type: str = "point",
        use_amp: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if model_wrapper is not None:
            self.model_wrapper = model_wrapper
        else:
            self.model_wrapper = SAM2ModelWrapper(model_id=model_id, device=device)

        self.device = self.model_wrapper.device
        self.model = self.model_wrapper.model
        self.processor = self.model_wrapper.processor
        self.freeze_vision_encoder = freeze_vision_encoder
        self.prompt_type = prompt_type
        self.use_amp = use_amp

        if self.freeze_vision_encoder:
            self._freeze_encoder()

        # Trainable parameters
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = CombinedSegmentationLoss()

    def _freeze_encoder(self) -> None:
        """Freeze vision backbone parameters to enable fast, memory-efficient fine-tuning."""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            if "image_encoder" in name or "vision_encoder" in name or "backbone" in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1

        logger.info(
            f"Vision backbone frozen: {frozen_count} tensors frozen, {trainable_count} tensors trainable."
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function supporting both multi-mask and single-mask crops with variable object counts."""
        images = torch.stack([b["image"] for b in batch])
        pil_images = [b["pil_image"] for b in batch]

        # Multi-object crops with plural 'masks'
        if "masks" in batch[0]:
            num_objs = [b["masks"].shape[0] for b in batch]
            h, w = batch[0]["masks"].shape[-2:]
            max_m = max(num_objs)

            padded_masks = torch.zeros((len(batch), max_m, 1, h, w), dtype=torch.float32)
            valid_mask = torch.zeros((len(batch), max_m), dtype=torch.bool)
            batch_points = []
            batch_labels = []
            batch_boxes = []

            for i, b in enumerate(batch):
                m_i = num_objs[i]
                padded_masks[i, :m_i] = b["masks"]
                valid_mask[i, :m_i] = True

                pts = b["points"].tolist()        # list of m_i items, each [[x, y]]
                lbls = b["point_labels"].tolist() # list of m_i items, each [label]

                # Explicitly pad points to max_m
                while len(pts) < max_m:
                    pts.append([[-10.0, -10.0]])
                    lbls.append([-10])

                box_list = b["boxes"].tolist()     # list of m_i items, each [x1, y1, x2, y2]
                while len(box_list) < max_m:
                    box_list.append([0.0, 0.0, 1.0, 1.0])

                batch_points.append(pts)
                batch_labels.append(lbls)
                batch_boxes.append(box_list)

            return {
                "image": images,
                "pil_image": pil_images,
                "masks": padded_masks,
                "valid_mask": valid_mask,
                "points": batch_points,
                "point_labels": batch_labels,
                "boxes": batch_boxes,
            }
        else:
            # Fallback for single-object legacy samples
            return {
                "image": images,
                "pil_image": pil_images,
                "masks": torch.stack([b["mask"] for b in batch]).unsqueeze(1),
                "valid_mask": torch.ones((len(batch), 1), dtype=torch.bool),
                "boxes": [[b["box"].tolist()] for b in batch],
                "points": [[[b["point"].squeeze(0).tolist()]] for b in batch],
                "point_labels": [[[int(b["point_label"].item())]] for b in batch],
            }

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Execute one training epoch with optional mixed precision (bfloat16 on A100)."""
        self.model.train()
        total_loss = 0.0
        steps = 0

        # Autocast setup: bfloat16 on NVIDIA A100/CUDA, float32 on MPS/CPU
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_cuda_amp = self.use_amp and (self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_cuda_amp and torch.cuda.is_bf16_supported()) else torch.float16

        for batch in dataloader:
            images = batch["pil_image"]
            target_masks = batch["masks"].to(self.device)  # (B, M_max, 1, H, W)
            valid_mask = batch["valid_mask"].to(self.device) # (B, M_max)

            if self.prompt_type == "box":
                proc_inputs = self.processor(
                    images=images,
                    input_boxes=batch["boxes"],
                    return_tensors="pt",
                ).to(self.device)
            elif self.prompt_type == "both":
                proc_inputs = self.processor(
                    images=images,
                    input_boxes=batch["boxes"],
                    input_points=batch["points"],
                    input_labels=batch["point_labels"],
                    return_tensors="pt",
                ).to(self.device)
            else:
                proc_inputs = self.processor(
                    images=images,
                    input_points=batch["points"],
                    input_labels=batch["point_labels"],
                    return_tensors="pt",
                ).to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda_amp):
                outputs = self.model(**proc_inputs)
                pred_logits = outputs.pred_masks[:, :, 0:1, :, :]  # (B, M_max, 1, 256, 256)

                # Filter valid objects across the batch
                valid_preds = pred_logits[valid_mask]      # (Total_Valid_M, 1, 256, 256)
                valid_targets = target_masks[valid_mask]  # (Total_Valid_M, 1, H, W)

                if valid_preds.shape[-2:] != valid_targets.shape[-2:]:
                    valid_preds = nn.functional.interpolate(
                        valid_preds,
                        size=valid_targets.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = self.criterion(valid_preds.float(), valid_targets.float())

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

        return total_loss / max(1, steps)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate on validation dataset.
        Returns (mean_val_loss, mean_val_iou).
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        steps = 0

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        use_cuda_amp = self.use_amp and (self.device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_cuda_amp and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.no_grad():
            for batch in dataloader:
                images = batch["pil_image"]
                target_masks = batch["masks"].to(self.device)
                valid_mask = batch["valid_mask"].to(self.device)

                if self.prompt_type == "box":
                    proc_inputs = self.processor(
                        images=images,
                        input_boxes=batch["boxes"],
                        return_tensors="pt",
                    ).to(self.device)
                elif self.prompt_type == "both":
                    proc_inputs = self.processor(
                        images=images,
                        input_boxes=batch["boxes"],
                        input_points=batch["points"],
                        input_labels=batch["point_labels"],
                        return_tensors="pt",
                    ).to(self.device)
                else:
                    proc_inputs = self.processor(
                        images=images,
                        input_points=batch["points"],
                        input_labels=batch["point_labels"],
                        return_tensors="pt",
                    ).to(self.device)

                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda_amp):
                    outputs = self.model(**proc_inputs)
                    pred_logits = outputs.pred_masks[:, :, 0:1, :, :]

                    valid_preds = pred_logits[valid_mask]
                    valid_targets = target_masks[valid_mask]

                    if valid_preds.shape[-2:] != valid_targets.shape[-2:]:
                        valid_preds = nn.functional.interpolate(
                            valid_preds,
                            size=valid_targets.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    loss = self.criterion(valid_preds.float(), valid_targets.float())
                total_loss += loss.item()

                pred_binary = (valid_preds > 0.0).float()
                target_binary = (valid_targets > 0.0).float()
                intersection = (pred_binary * target_binary).sum(dim=(-1, -2))
                union = pred_binary.sum(dim=(-1, -2)) + target_binary.sum(dim=(-1, -2)) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.mean().item()
                steps += 1

        return total_loss / max(1, steps), total_iou / max(1, steps)

    def train(
        self,
        dataset: AstronomicalDataset,
        epochs: int = 5,
        batch_size: int = 2,
        val_dataset: Optional[AstronomicalDataset] = None,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete fine-tuning routine.

        Parameters
        ----------
        dataset : AstronomicalDataset
            Training dataset.
        epochs : int, default 5
            Number of training epochs.
        batch_size : int, default 2
            Batch size of cropped images.
        val_dataset : AstronomicalDataset, optional
            Optional validation dataset.
        output_dir : str or Path, optional
            Directory to save checkpoints.
        verbose : bool, default True
            Print progress and metrics per epoch.
        """
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
            if val_dataset is not None
            else None
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
        }

        best_val_loss = float("inf")

        if verbose:
            print(f"[sam2roman] Starting fine-tuning for {epochs} epochs on {self.device}...")
            print(f"  -> Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")
            is_multi = getattr(dataset, "all_masks_in_crop", getattr(getattr(dataset, "dataset", None), "all_masks_in_crop", False))
            multi_str = " (all masks in crop)" if is_multi else " (1 mask per crop)"
            print(f"  -> Training crops:       {len(dataset)}{multi_str}")
            if val_dataset:
                is_val_multi = getattr(val_dataset, "all_masks_in_crop", getattr(getattr(val_dataset, "dataset", None), "all_masks_in_crop", False))
                val_multi_str = " (all masks in crop)" if is_val_multi else " (1 mask per crop)"
                print(f"  -> Validation crops:     {len(val_dataset)}{val_multi_str}")
            print(f"  -> Prompt mode:          {self.prompt_type}")
            print("-" * 65)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            elapsed = time.time() - t0

            val_str = ""
            if val_loader is not None:
                val_loss, val_iou = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_iou"].append(val_iou)
                val_str = f" | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}"

                if output_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = Path(output_dir) / "checkpoint_best.pt"
                    self.save_checkpoint(best_path)

            if verbose:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}{val_str} ({elapsed:.1f}s)")

            if output_dir:
                save_path = Path(output_dir) / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(save_path)

        if verbose:
            print("-" * 65)
            print("[sam2roman] Fine-tuning completed successfully!")

        return history

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_id": self.model_wrapper.model_id,
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model weights from checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        logger.info(f"Loaded checkpoint from: {path}")
