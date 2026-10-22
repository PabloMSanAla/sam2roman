"""
Training and fine-tuning engine for SAM 2 on astronomical datasets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union

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

        if self.freeze_vision_encoder:
            self._freeze_encoder()

        # Gather trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = CombinedSegmentationLoss()

    def _freeze_encoder(self) -> None:
        """Freeze vision backbone parameters to enable fast, memory-efficient fine-tuning."""
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            # Freeze vision encoder / backbone
            if "image_encoder" in name or "vision_encoder" in name or "backbone" in name:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
                trainable_count += 1

        logger.info(
            f"Encoder frozen: {frozen_count} tensors frozen, {trainable_count} tensors trainable."
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Execute one training epoch.

        Parameters
        ----------
        dataloader : DataLoader
            Batch provider for AstronomicalDataset.

        Returns
        -------
        float
            Mean epoch loss.
        """
        self.model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
        for batch in pbar:
            images = batch["pil_image"]
            target_masks = batch["mask"].to(self.device)  # (B, 1, H, W)
            points = batch["point"].unsqueeze(1).tolist()  # (B, 1, 1, 2)
            labels = batch["point_label"].unsqueeze(1).tolist()

            try:
                inputs = self.processor(
                    images=images,
                    input_points=points,
                    input_labels=labels,
                    return_tensors="pt",
                ).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)

                pred_logits = outputs.pred_masks[:, 0, 0:1, :, :]  # Leading prompt mask logit
                # Resize pred_logits to match target mask dimensions if necessary
                if pred_logits.shape[-2:] != target_masks.shape[-2:]:
                    pred_logits = nn.functional.interpolate(
                        pred_logits,
                        size=target_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss = self.criterion(pred_logits, target_masks)
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                total_loss += loss_val
                steps += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            except Exception as e:
                logger.warning(f"Error in training step: {e}")
                continue

        return total_loss / max(1, steps)

    def train(
        self,
        dataset: AstronomicalDataset,
        epochs: int = 5,
        batch_size: int = 2,
        val_dataset: Optional[AstronomicalDataset] = None,
        output_dir: Optional[Union[str, Path]] = None,
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
            Batch size.
        val_dataset : AstronomicalDataset, optional
            Optional validation dataset.
        output_dir : str or Path, optional
            Directory to save checkpoints.
        """
        def collate_fn(batch):
            return {
                "image": torch.stack([b["image"] for b in batch]),
                "pil_image": [b["pil_image"] for b in batch],
                "mask": torch.stack([b["mask"] for b in batch]),
                "box": torch.stack([b["box"] for b in batch]),
                "point": torch.stack([b["point"] for b in batch]),
                "point_label": torch.stack([b["point_label"] for b in batch]),
            }

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        history = {"train_loss": []}

        logger.info(f"Starting fine-tuning for {epochs} epochs on {self.device}...")
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(dataloader)
            history["train_loss"].append(avg_loss)
            logger.info(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

            if output_dir:
                save_path = Path(output_dir) / f"checkpoint_epoch_{epoch}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                self.save_checkpoint(save_path)

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
        logger.info(f"Checkpoint saved to: {path}")
