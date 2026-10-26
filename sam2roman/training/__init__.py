"""Training and fine-tuning pipelines for SAM 2 on astronomical data."""

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss
from sam2roman.training.trainer import SAM2Trainer

__all__ = ["AstronomicalDataset", "DiceLoss", "CombinedSegmentationLoss", "SAM2Trainer"]
