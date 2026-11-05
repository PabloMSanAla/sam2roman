"""Training and fine-tuning pipelines for SAM 2 and UNet++ on astronomical data."""

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.training.unetpp_trainer import UNetPPTrainer
from sam2roman.training.steer_query_trainer import SteerQueryTrainer
from sam2roman.losses.steer_query_loss import (
    SteerQueryLoss,
    build_steer_query_optimizer,
    render_star_center_heatmap,
    compute_spike_over_galaxy_weights,
)
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
    "UNetPPTrainer",
    "SteerQueryTrainer",
    "SteerQueryLoss",
    "build_steer_query_optimizer",
    "render_star_center_heatmap",
    "compute_spike_over_galaxy_weights",
    "dilate_binary_mask",
    "compute_relaxed_boundary_iou",
    "compute_soft_continuous_iou",
    "evaluate_batch_metrics",
]
