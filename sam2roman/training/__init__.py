"""Training and fine-tuning pipelines for SAM 2 on astronomical data."""

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.training.metrics import (
    dilate_binary_mask,
    compute_relaxed_boundary_iou,
    compute_soft_continuous_iou,
    evaluate_batch_metrics,
)

__all__ = [
    "AstronomicalDataset",
    "DiceLoss",
    "CombinedSegmentationLoss",
    "SAM2Trainer",
    "dilate_binary_mask",
    "compute_relaxed_boundary_iou",
    "compute_soft_continuous_iou",
    "evaluate_batch_metrics",
]
