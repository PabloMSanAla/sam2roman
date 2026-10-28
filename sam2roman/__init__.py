"""
sam2roman: SAM 2 inference and training pipeline for Roman Space Telescope and astronomical FITS imagery.
"""

import os
# Safeguard for macOS dual-libomp conflict between conda-forge and pip
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sam2roman.io.fits import FITSImage, stretch_astro_image
from sam2roman.models.sam2 import SAM2ModelWrapper
from sam2roman.inference.segmentor import SAM2Segmentor
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.visualization.stretch import leakySB, leakySB_inverse
from sam2roman.visualization.display import plot_masks_overlay, plot_detection_summary
from sam2roman.utils.synthetic import generate_synthetic_roman_fits

__version__ = "0.1.0"
__all__ = [
    "FITSImage",
    "stretch_astro_image",
    "SAM2ModelWrapper",
    "SAM2Segmentor",
    "AstronomicalDataset",
    "DiceLoss",
    "CombinedSegmentationLoss",
    "SAM2Trainer",
    "leakySB",
    "leakySB_inverse",
    "plot_masks_overlay",
    "plot_detection_summary",
    "generate_synthetic_roman_fits",
]
