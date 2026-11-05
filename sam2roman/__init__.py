"""
sam2roman: SAM 2 inference and training pipeline for Roman Space Telescope and astronomical FITS imagery.
"""

import os
# Safeguard for macOS dual-libomp conflict between conda-forge and pip
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sam2roman.io.fits import FITSImage, stretch_astro_image
from sam2roman.io.asdf_io import ASDFImage
from sam2roman.io.reader import load_astro_image
from sam2roman.models.sam2 import SAM2ModelWrapper
from sam2roman.models.unetpp import UNetPlusPlus
from sam2roman.models.steered_strip_conv import LearnableSteeredStripConv
from sam2roman.models.steer_query_head import SteerQueryHead, UNetPlusPlusSteerQuery
from sam2roman.inference.segmentor import SAM2Segmentor
from sam2roman.inference.unetpp_segmentor import UNetPPSegmentor
from sam2roman.inference.steer_query_segmentor import SteerQuerySegmentor
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.losses import DiceLoss, CombinedSegmentationLoss
from sam2roman.losses.steer_query_loss import (
    SteerQueryLoss,
    build_steer_query_optimizer,
    render_star_center_heatmap,
    compute_spike_over_galaxy_weights,
)
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.training.unetpp_trainer import UNetPPTrainer
from sam2roman.training.steer_query_trainer import SteerQueryTrainer
from sam2roman.visualization.stretch import leakySB, leakySB_inverse
from sam2roman.visualization.display import (
    plot_masks_overlay,
    plot_detection_summary,
    plot_two_panel_prediction,
)
from sam2roman.utils.synthetic import generate_synthetic_roman_fits

__version__ = "0.1.0"
__all__ = [
    "FITSImage",
    "ASDFImage",
    "load_astro_image",
    "stretch_astro_image",
    "SAM2ModelWrapper",
    "UNetPlusPlus",
    "LearnableSteeredStripConv",
    "SteerQueryHead",
    "UNetPlusPlusSteerQuery",
    "SAM2Segmentor",
    "UNetPPSegmentor",
    "SteerQuerySegmentor",
    "AstronomicalDataset",
    "DiceLoss",
    "CombinedSegmentationLoss",
    "SteerQueryLoss",
    "build_steer_query_optimizer",
    "render_star_center_heatmap",
    "compute_spike_over_galaxy_weights",
    "SAM2Trainer",
    "UNetPPTrainer",
    "SteerQueryTrainer",
    "leakySB",
    "leakySB_inverse",
    "plot_masks_overlay",
    "plot_detection_summary",
    "plot_two_panel_prediction",
    "generate_synthetic_roman_fits",
]
