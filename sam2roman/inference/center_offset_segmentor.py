"""
Inference engine and instance deblender for Center Offset Voting UNet++.
========================================================================

Converts multi-task predictions:
1. Semantic foreground probability
2. Object center keypoint heatmap
3. Dense 2D offset vector field (delta_y, delta_x)
into discrete astronomical source instances with celestial WCS coordinates and photometry.

Includes colinear ray-aligned voting to deblend long diffraction spikes from intersecting galaxies.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from PIL import Image

from sam2roman.io.fits import FITSImage, stretch_astro_image
from sam2roman.io.asdf_io import ASDFImage
from sam2roman.io.reader import load_astro_image
from sam2roman.models.unetpp_offset import UNetPlusPlusCenterOffset
from sam2roman.inference.segmentor import SegmentationResult
from sam2roman.training.offset_metrics import extract_center_peaks

logger = logging.getLogger(__name__)


class CenterOffsetSegmentor:
    """
    Inference segmentor using Center Offset Voting and Criss-Cross Attention.

    Parameters
    ----------
    model : UNetPlusPlusCenterOffset, optional
        Pre-configured model.
    checkpoint_path : str or Path, optional
        Path to saved checkpoint (e.g. 'checkpoints/center_offset_best.pt').
    device : str or torch.device, optional
        Target hardware device. Auto-detected if None.
    """

    def __init__(
        self,
        model: Optional[UNetPlusPlusCenterOffset] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        in_channels: int = 3,
        semantic_classes: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        use_cca: bool = True,
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
            self.model = UNetPlusPlusCenterOffset(
                in_channels=in_channels,
                semantic_classes=semantic_classes,
                features=features,
                use_cca=use_cca,
            ).to(self.device)

        self.model.eval()

    def _load_from_checkpoint(self, path: Union[str, Path]) -> UNetPlusPlusCenterOffset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device)
        in_channels = ckpt.get("in_channels", 3)
        semantic_classes = ckpt.get("semantic_classes", 1)
        features = tuple(ckpt.get("features", (32, 64, 128, 256, 512)))
        use_cca = ckpt.get("use_cca", True)

        model = UNetPlusPlusCenterOffset(
            in_channels=in_channels,
            semantic_classes=semantic_classes,
            features=features,
            use_cca=use_cca,
        ).to(self.device)

        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded Center Offset UNet++ weights from: {path}")
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "CenterOffsetSegmentor":
        return cls(checkpoint_path=checkpoint_path, device=device)

    @staticmethod
    def deblend_from_maps(
        pred_semantic: np.ndarray,
        pred_center: np.ndarray,
        pred_offsets: np.ndarray,
        fg_thresh: float = 0.50,
        center_thresh: float = 0.30,
        min_distance: int = 3,
        colinear_thresh: float = 0.70,
        max_vote_dist: float = 64.0,
        min_area: int = 4,
        max_area: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[float]], List[float], List[Tuple[float, float]]]:
        """
        Deblend instances directly from predicted semantic, center, and offset maps.

        Parameters
        ----------
        pred_semantic : np.ndarray
            Foreground probability array of shape (H, W).
        pred_center : np.ndarray
            Center keypoint heatmap of shape (H, W).
        pred_offsets : np.ndarray
            Offset vector field of shape (2, H, W) where [0] is dy and [1] is dx.
        fg_thresh : float, default 0.50
        center_thresh : float, default 0.30
        min_distance : int, default 3
        colinear_thresh : float, default 0.70
        max_vote_dist : float, default 64.0
        min_area : int, default 4
        max_area : int, optional

        Returns
        -------
        Tuple containing:
            - labeled_map: (H, W) integer map of instance IDs (0: bg, 1..N: instances)
            - masks: List of boolean masks
            - boxes: List of [xmin, ymin, xmax, ymax]
            - scores: List of mean foreground scores
            - centroids: List of (cx, cy)
        """
        p_fg = pred_semantic.squeeze()
        p_ctr = pred_center.squeeze()
        h, w = p_fg.shape

        # 1. Extract candidate centers via peak NMS
        peaks = extract_center_peaks(p_ctr, threshold=center_thresh, min_distance=min_distance)

        # 2. Foreground mask
        fg_mask = (p_fg >= fg_thresh)
        labeled_map = np.zeros((h, w), dtype=np.int32)

        if len(peaks) > 0 and np.any(fg_mask):
            centers = np.array([[p[0], p[1]] for p in peaks], dtype=np.float32)  # (K, 2): [cy, cx]
            fg_ys, fg_xs = np.where(fg_mask)
            dy_pred = pred_offsets[0, fg_ys, fg_xs]
            dx_pred = pred_offsets[1, fg_ys, fg_xs]

            # Shifted destination points (y*, x*)
            dest_ys = fg_ys + dy_pred
            dest_xs = fg_xs + dx_pred

            # Distance matrix between destination points and center peaks
            dists = np.sqrt(
                (dest_ys[:, None] - centers[None, :, 0]) ** 2
                + (dest_xs[:, None] - centers[None, :, 1]) ** 2
            )

            # Colinear angular check for distant points
            vec_to_centers_y = centers[None, :, 0] - fg_ys[:, None]
            vec_to_centers_x = centers[None, :, 1] - fg_xs[:, None]
            norm_c = np.sqrt(vec_to_centers_y ** 2 + vec_to_centers_x ** 2) + 1e-6
            norm_p = np.sqrt(dy_pred ** 2 + dx_pred ** 2)[:, None] + 1e-6

            cos_angles = (dy_pred[:, None] * vec_to_centers_y + dx_pred[:, None] * vec_to_centers_x) / (norm_p * norm_c)

            is_distant = (norm_c > 10.0)
            misaligned = is_distant & (cos_angles < colinear_thresh)
            dists[misaligned] += 1000.0

            best_center_indices = np.argmin(dists, axis=1)
            min_dists = dists[np.arange(len(fg_ys)), best_center_indices]

            valid_assignment = min_dists <= max_vote_dist
            assigned_labels = best_center_indices[valid_assignment] + 1
            labeled_map[fg_ys[valid_assignment], fg_xs[valid_assignment]] = assigned_labels

        # 3. Extract individual instance masks & properties
        masks_list: List[np.ndarray] = []
        boxes_list: List[List[float]] = []
        scores_list: List[float] = []
        areas_list: List[int] = []
        centroids_list: List[Tuple[float, float]] = []

        unique_instances = np.unique(labeled_map)
        unique_instances = unique_instances[unique_instances > 0]

        for obj_id in unique_instances:
            m = (labeled_map == obj_id)
            area = int(np.sum(m))
            if area < min_area or (max_area is not None and area > max_area):
                continue

            ys, xs = np.where(m)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            score = float(np.mean(p_fg[m]))

            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))

            masks_list.append(m)
            boxes_list.append([x_min, y_min, x_max, y_max])
            scores_list.append(score)
            areas_list.append(area)
            centroids_list.append((cx, cy))

        return labeled_map, masks_list, boxes_list, scores_list, areas_list, centroids_list

    def segment_image(
        self,
        image: Union[FITSImage, ASDFImage, np.ndarray, str, Path],
        stretch_type: str = "leakysb",
        fg_thresh: float = 0.50,
        center_thresh: float = 0.30,
        min_distance: int = 3,
        colinear_thresh: float = 0.70,
        max_vote_dist: float = 64.0,
        min_area: int = 4,
        max_area: Optional[int] = None,
        verbose: bool = True,
    ) -> SegmentationResult:
        """
        Segment all sources in image using Center Offset Voting.

        Parameters
        ----------
        image : astronomical image or filepath
        stretch_type : str, default 'leakysb'
        fg_thresh : float, default 0.50
            Threshold on foreground semantic probability.
        center_thresh : float, default 0.30
            Threshold for detecting center peaks in heatmap.
        min_distance : int, default 3
            Footprint radius for local peak NMS.
        colinear_thresh : float, default 0.70
            Cosine angular threshold for diffraction spike assignment.
        max_vote_dist : float, default 64.0
            Maximum residual Euclidean distance for offset clustering.
        min_area : int, default 4
            Minimum pixels to retain an instance.
        max_area : int, optional
            Maximum area to filter halos.
        verbose : bool, default True

        Returns
        -------
        SegmentationResult:
            Discretely deblended instances with masks, boxes, scores, and WCS celestial coordinates.
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

        # Convert to tensor matching model input channels
        if in_channels == 1:
            input_tensor = torch.from_numpy(raw_data).unsqueeze(0).unsqueeze(0).float().to(self.device)
        else:
            rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
            input_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Single forward pass
        t_fwd = time.time()
        with torch.no_grad():
            preds = self.model(input_tensor)
            p_fg = torch.sigmoid(preds["semantic"][0, 0]).float().cpu().numpy()
            p_ctr = torch.sigmoid(preds["center"][0, 0]).float().cpu().numpy()
            pred_offsets = preds["offset"][0].float().cpu().numpy()  # (2, H, W)
        fwd_time = time.time() - t_fwd

        # Deblend instances from predictions
        labeled_map, masks_list, boxes_list, scores_list, areas_list, centroids_list = self.deblend_from_maps(
            pred_semantic=p_fg,
            pred_center=p_ctr,
            pred_offsets=pred_offsets,
            fg_thresh=fg_thresh,
            center_thresh=center_thresh,
            min_distance=min_distance,
            colinear_thresh=colinear_thresh,
            max_vote_dist=max_vote_dist,
            min_area=min_area,
            max_area=max_area,
        )

        sky_centroids: Optional[List[Tuple[float, float]]] = [] if astro_obj and astro_obj.wcs else None
        if sky_centroids is not None and astro_obj is not None:
            for cx, cy in centroids_list:
                sky_pt = astro_obj.pixel_to_sky(cx, cy)
                if sky_pt:
                    sky_centroids.append(sky_pt)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"[sam2roman] Center Offset Voting segmented ({w}x{h}) in {elapsed*1000:.1f}ms "
                f"(forward pass: {fwd_time*1000:.1f}ms on {self.device}). Deblended {len(masks_list)} sources."
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
