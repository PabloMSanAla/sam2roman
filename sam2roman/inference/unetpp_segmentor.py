"""
Inference engine and instance deblender for boundary-aware UNet++.
Converts dense 3-class predictions (Background, Interior, Boundary) into discrete
astronomical source instances with celestial WCS coordinates and photometry.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from PIL import Image

from sam2roman.io.fits import FITSImage
from sam2roman.models.unetpp import UNetPlusPlus
from sam2roman.inference.segmentor import SegmentationResult

logger = logging.getLogger(__name__)


class UNetPPSegmentor:
    """
    Inference segmentor for boundary-aware UNet++ models on astronomical FITS images.

    Executes a single-pass forward pass to produce dense 3-class probability maps
    (Sky, Interior, Edge) and applies fast marker-controlled watershed deblending
    to separate overlapping or crowded stars and galaxies into discrete instances.

    Parameters
    ----------
    model : UNetPlusPlus, optional
        Pre-configured UNet++ model.
    checkpoint_path : str or Path, optional
        Path to fine-tuned checkpoint (e.g. 'checkpoints/unetpp_best.pt').
    device : str or torch.device, optional
        Target device ('cuda', 'mps', 'cpu'). Auto-detected if None.
    """

    def __init__(
        self,
        model: Optional[UNetPlusPlus] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
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
            self.model = UNetPlusPlus(
                in_channels=in_channels,
                num_classes=num_classes,
                features=features,
            ).to(self.device)

        self.model.eval()

    def _load_from_checkpoint(self, path: Union[str, Path]) -> UNetPlusPlus:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device)
        in_channels = ckpt.get("in_channels", 3)
        num_classes = ckpt.get("num_classes", 3)
        features = tuple(ckpt.get("features", (32, 64, 128, 256, 512)))
        deep_supervision = ckpt.get("deep_supervision", False)

        model = UNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            deep_supervision=deep_supervision,
        ).to(self.device)

        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded UNet++ weights from checkpoint: {path}")
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "UNetPPSegmentor":
        """Load UNetPPSegmentor directly from a saved checkpoint."""
        return cls(checkpoint_path=checkpoint_path, device=device)

    def segment_image(
        self,
        image: Union[FITSImage, np.ndarray, str, Path],
        stretch_type: str = "leakysb",
        interior_thresh: float = 0.45,
        fg_thresh: float = 0.50,
        min_area: int = 10,
        max_area: Optional[int] = None,
        verbose: bool = True,
    ) -> SegmentationResult:
        """
        Segment all astronomical sources in the image using the fine-tuned UNet++.

        Parameters
        ----------
        image : FITSImage, np.ndarray, str, or Path
            Input astronomical image.
        stretch_type : str, default 'leakysb'
            Intensity transformation.
        interior_thresh : float, default 0.45
            Probability threshold on source interiors to identify core seeds.
        fg_thresh : float, default 0.50
            Threshold on total foreground probability (1 - P_bg).
        min_area : int, default 10
            Minimum area in pixels to retain a detected source.
        max_area : int, optional
            Maximum area to filter background halos.
        verbose : bool, default True
            Print execution timing and detection count.

        Returns
        -------
        SegmentationResult
            Extracted source instances with masks, boxes, scores, centroids, and photometry support.
        """
        t0 = time.time()

        # Ingest image
        fits_obj: Optional[FITSImage] = None
        if isinstance(image, (str, Path)):
            fits_obj = FITSImage.from_file(image)
            raw_data = fits_obj.data
            pil_img = fits_obj.to_pil(stretch_type=stretch_type)
        elif isinstance(image, FITSImage):
            fits_obj = image
            raw_data = fits_obj.data
            pil_img = fits_obj.to_pil(stretch_type=stretch_type)
        elif isinstance(image, np.ndarray):
            raw_data = image
            fits_obj = FITSImage(image)
            pil_img = fits_obj.to_pil(stretch_type=stretch_type)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        h, w = raw_data.shape

        # Convert to tensor (1, 3, H, W)
        rgb_arr = np.array(pil_img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Single forward pass
        t_fwd = time.time()
        with torch.no_grad():
            logits = self.model(input_tensor)
            if isinstance(logits, list):
                logits = logits[-1]
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (3, H, W)

        fwd_time = time.time() - t_fwd

        p_bg = probs[0]
        p_int = probs[1]
        p_edge = probs[2]

        # 1. Detect interior cores (seed markers)
        core_mask = (p_int >= interior_thresh)
        markers, num_cores = ndimage.label(core_mask)

        # 2. Total foreground region
        fg_mask = (p_bg < fg_thresh) | (p_int + p_edge >= fg_thresh)

        # 3. Fast distance-transform watershed expansion
        if num_cores > 0 and np.any(fg_mask):
            dist = ndimage.distance_transform_edt(fg_mask)
            dist_int = (np.max(dist) - dist).astype(np.uint8)
            labeled_map = ndimage.watershed_ift(dist_int, markers.astype(np.int32))
            labeled_map[~fg_mask] = 0
        else:
            labeled_map = np.zeros((h, w), dtype=np.int32)

        # 4. Extract individual instance masks
        masks_list: List[np.ndarray] = []
        boxes_list: List[List[float]] = []
        scores_list: List[float] = []
        areas_list: List[int] = []
        centroids_list: List[Tuple[float, float]] = []
        sky_centroids: Optional[List[Tuple[float, float]]] = [] if fits_obj and fits_obj.wcs else None

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

            # Confidence score: mean foreground probability over the object
            score = float(np.mean(1.0 - p_bg[m]))

            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))

            masks_list.append(m)
            boxes_list.append([x_min, y_min, x_max, y_max])
            scores_list.append(score)
            areas_list.append(area)
            centroids_list.append((cx, cy))

            if sky_centroids is not None:
                sky_pt = fits_obj.pixel_to_sky(cx, cy)
                if sky_pt:
                    sky_centroids.append(sky_pt)

        elapsed = time.time() - t0

        if verbose:
            print(f"[sam2roman] UNet++ segmented ({w}x{h}) in {elapsed*1000:.1f}ms "
                  f"(forward pass: {fwd_time*1000:.1f}ms on {self.device}). Extracted {len(masks_list)} sources.")

        return SegmentationResult(
            masks=masks_list,
            boxes=boxes_list,
            scores=scores_list,
            areas=areas_list,
            centroids=centroids_list,
            sky_centroids=sky_centroids,
            image_shape=(h, w),
        )
