"""
Inference engine and star deblender for SteerQuery UNet++.
==========================================================

Runs end-to-end GPU instance segmentation for stars with long diffraction spikes:
1. Passes Roman WFI flux through UNetPlusPlusSteerQuery.
2. Directly outputs integer instance_mask (B, H, W) on GPU without CPU voting or clustering.
3. Converts unique star instances (1..N) into discrete masks, boxes, scores, and WCS coordinates.
4. Galaxies remain strictly background (ID = 0).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from sam2roman.io.fits import FITSImage, stretch_astro_image
from sam2roman.io.asdf_io import ASDFImage
from sam2roman.io.reader import load_astro_image
from sam2roman.models.steer_query_head import UNetPlusPlusSteerQuery
from sam2roman.inference.segmentor import SegmentationResult

logger = logging.getLogger(__name__)


class SteerQuerySegmentor:
    """
    Inference segmentor using SteerQuery direct GPU instance segmentation.

    Parameters
    ----------
    model : UNetPlusPlusSteerQuery, optional
        Pre-instantiated model.
    checkpoint_path : str or Path, optional
        Path to saved checkpoint (e.g. 'checkpoints/steer_query_best.pt').
    in_channels : int, default 3
    num_classes : int, default 3
    features : tuple of int, default (32, 64, 128, 256, 512)
    device : str or torch.device, optional
    """

    def __init__(
        self,
        model: Optional[UNetPlusPlusSteerQuery] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        proto_dim: int = 32,
        N_max: int = 128,
        tau_mask: float = 0.35,
        tau_star: float = 0.30,
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

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model = self._load_from_checkpoint(checkpoint_path)
        else:
            self.model = UNetPlusPlusSteerQuery(
                in_channels=in_channels,
                num_classes=num_classes,
                features=features,
                proto_dim=proto_dim,
                N_max=N_max,
                tau_mask=tau_mask,
                tau_star=tau_star,
            ).to(self.device)

        self.model.eval()

    def _load_from_checkpoint(self, path: Union[str, Path]) -> UNetPlusPlusSteerQuery:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device)
        in_channels = ckpt.get("in_channels", 3)
        num_classes = ckpt.get("num_classes", 3)
        features = tuple(ckpt.get("features", (32, 64, 128, 256, 512)))
        proto_dim = ckpt.get("proto_dim", 32)
        N_max = ckpt.get("N_max", 128)

        model = UNetPlusPlusSteerQuery(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            proto_dim=proto_dim,
            N_max=N_max,
        ).to(self.device)

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded SteerQuery UNet++ weights from: {path}")
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "SteerQuerySegmentor":
        return cls(checkpoint_path=checkpoint_path, device=device)

    def segment_image(
        self,
        image: Union[FITSImage, ASDFImage, np.ndarray, str, Path],
        stretch_type: str = "leakysb",
        min_area: int = 4,
        max_area: Optional[int] = None,
        verbose: bool = True,
    ) -> SegmentationResult:
        """
        Segment all star instances in an image directly on GPU.

        Parameters
        ----------
        image : astronomical image or filepath
        stretch_type : str, default 'leakysb'
        min_area : int, default 4
        max_area : int, optional
        verbose : bool, default True

        Returns
        -------
        SegmentationResult:
            Deblended star instances with integer masks, boxes, scores, and WCS celestial coordinates.
        """
        t0 = time.time()

        # Ingest image
        astro_obj: Optional[Union[FITSImage, ASDFImage]] = None
        if isinstance(image, (str, Path)):
            astro_obj = load_astro_image(image)
            raw_data = astro_obj.data
            pil_img = astro_obj.to_pil(stretch_type=stretch_type)
        elif isinstance(image, (FITSImage, ASDFImage)):
            astro_obj = image
            raw_data = astro_obj.data
            pil_img = astro_obj.to_pil(stretch_type=stretch_type)
        elif isinstance(image, np.ndarray):
            raw_data = image
            astro_obj = FITSImage(image)
            pil_img = astro_obj.to_pil(stretch_type=stretch_type)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        h, w = raw_data.shape
        in_channels = getattr(self.model, "in_channels", 3)

        if in_channels == 1:
            input_tensor = torch.from_numpy(raw_data).unsqueeze(0).unsqueeze(0).float().to(self.device)
        else:
            rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
            input_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Single GPU forward pass
        t_fwd = time.time()
        with torch.no_grad():
            preds = self.model(input_tensor)
            inst_mask_gpu = preds["instance_mask"][0]  # (H, W) torch.int32
            scores = preds["scores"][0].float().cpu().numpy()  # (N,)
            star_coords = preds["star_coords"][0].float().cpu().numpy()  # (N, 2)
        fwd_time = time.time() - t_fwd

        inst_mask_np = inst_mask_gpu.cpu().numpy()

        masks_list: List[np.ndarray] = []
        boxes_list: List[List[float]] = []
        scores_list: List[float] = []
        areas_list: List[int] = []
        centroids_list: List[Tuple[float, float]] = []
        sky_centroids: Optional[List[Tuple[float, float]]] = [] if astro_obj and astro_obj.wcs else None

        unique_ids = np.unique(inst_mask_np)
        unique_ids = unique_ids[unique_ids > 0]

        for inst_id in unique_ids:
            m = (inst_mask_np == inst_id)
            area = int(np.sum(m))
            if area < min_area or (max_area is not None and area > max_area):
                continue

            ys, xs = np.where(m)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            # Query score corresponding to instance ID (1-indexed)
            q_idx = int(inst_id) - 1
            sc = float(scores[q_idx]) if q_idx < len(scores) else 1.0

            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))

            masks_list.append(m)
            boxes_list.append([x_min, y_min, x_max, y_max])
            scores_list.append(sc)
            areas_list.append(area)
            centroids_list.append((cx, cy))

            if sky_centroids is not None and astro_obj is not None:
                sky_pt = astro_obj.pixel_to_sky(cx, cy)
                if sky_pt:
                    sky_centroids.append(sky_pt)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"[sam2roman] SteerQuery segmented ({w}x{h}) in {elapsed*1000:.1f}ms "
                f"(forward pass: {fwd_time*1000:.1f}ms on {self.device}). Found {len(masks_list)} stars."
            )

        return SegmentationResult(
            masks=masks_list,
            boxes=boxes_list,
            scores=scores_list,
            areas=areas_list,
            centroids=centroids_list,
            sky_centroids=sky_centroids,
            image_shape=(h, w),
        )
