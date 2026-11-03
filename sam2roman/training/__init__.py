"""Training and fine-tuning pipelines for SAM 2 and UNet++ on astronomical data."""

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.training.metrics import (
    dilate_binary_mask,
    compute_relaxed_boundary_iou,
    compute_soft_continuous_iou,
    evaluate_batch_metrics,
)
from sam2roman.training.offset_target import (
    compute_instance_centroids,
    generate_center_offset_targets_numpy,
    generate_center_offset_batch,
)
from sam2roman.training.offset_losses import (
    CenterFocalLoss,
    MaskedSmoothL1Loss,
    AngularCosineDistanceLoss,
    CenterOffsetCompositeLoss,
)
from sam2roman.training.offset_metrics import (
    extract_center_peaks,
    evaluate_center_detection,
    compute_offset_epe,
)

from sam2roman.training.center_offset_trainer import CenterOffsetTrainer

__all__ = [
    "AstronomicalDataset",
    "DiceLoss",
    "CombinedSegmentationLoss",
    "SAM2Trainer",
    "CenterOffsetTrainer",
    "dilate_binary_mask",
    "compute_relaxed_boundary_iou",
    "compute_soft_continuous_iou",
    "evaluate_batch_metrics",
    "compute_instance_centroids",
    "generate_center_offset_targets_numpy",
    "generate_center_offset_batch",
    "CenterFocalLoss",
    "MaskedSmoothL1Loss",
    "AngularCosineDistanceLoss",
    "CenterOffsetCompositeLoss",
    "extract_center_peaks",
    "evaluate_center_detection",
    "compute_offset_epe",
]
