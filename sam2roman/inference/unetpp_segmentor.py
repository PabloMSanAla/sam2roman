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

from sam2roman.io.fits import FITSImage, stretch_astro_image
from sam2roman.io.asdf_io import ASDFImage
from sam2roman.io.reader import load_astro_image
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
        image: Union[FITSImage, ASDFImage, np.ndarray, str, Path],
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
        image : FITSImage, ASDFImage, np.ndarray, str, or Path
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
        sky_centroids: Optional[List[Tuple[float, float]]] = [] if astro_obj and astro_obj.wcs else None

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

            if sky_centroids is not None and astro_obj is not None:
                sky_pt = astro_obj.pixel_to_sky(cx, cy)
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

    @staticmethod
    def _build_tile_window(
        size: int,
        overlap: int,
        is_top: bool,
        is_bottom: bool,
        is_left: bool,
        is_right: bool,
    ) -> np.ndarray:
        """
        Construct a 2D cosine blending window for sliding tile inference.
        Edges touching physical image boundaries maintain weight 1.0 to prevent border fading.
        """
        if overlap <= 0:
            return np.ones((size, size), dtype=np.float32)

        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32)))

        wy = np.ones(size, dtype=np.float32)
        if not is_top:
            wy[:overlap] = ramp
        if not is_bottom:
            wy[-overlap:] = ramp[::-1]

        wx = np.ones(size, dtype=np.float32)
        if not is_left:
            wx[:overlap] = ramp
        if not is_right:
            wx[-overlap:] = ramp[::-1]

        return np.outer(wy, wx)

    def predict_dense(
        self,
        image: Union[np.ndarray, FITSImage, ASDFImage, str, Path],
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 2,
        stretch_type: str = "leakysb",
        zero_points: float = 31.4,
        pixel_scale: float = 0.11,
        mmin: float = 13.0,
        mmax: float = 32.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run dense 3-class segmentation inference over the whole astronomical image.
        Seamlessly handles large images (e.g. 4088x4088 Roman WFI exposures) via
        tiled sliding-window inference with cosine window blending to avoid VRAM overflow.

        Parameters
        ----------
        image : np.ndarray, FITSImage, ASDFImage, str, or Path
            Input astronomical image data or file path (.fits, .fits.gz, .asdf).
        tile_size : int, default 512
            Tile width/height in pixels. Must be divisible by 16.
            Set to 0 to run a single unchunked forward pass across the entire image.
        overlap : int, default 64
            Overlap in pixels between adjacent tiles for seamless blending.
        batch_size : int, default 2
            Number of tiles processed concurrently in a forward batch on GPU.
        stretch_type : str, default 'leakysb'
            Intensity transformation ('leakysb', 'asinh', etc.).
        zero_points : float, default 31.4
            Photometric zero point for leakySB.
        pixel_scale : float, default 0.11
            Plate scale in arcseconds/pixel.
        mmin : float, default 13.0
            Bright magnitude limit for leakySB.
        mmax : float, default 32.0
            Faint magnitude limit for leakySB.
        verbose : bool, default True
            Log execution time and tile breakdown.

        Returns
        -------
        pred_mask : np.ndarray
            Discrete 2D integer mask (H, W) where:
            0 = Sky Background, 1 = Source Interior, 2 = Deblending Boundary.
        probs : np.ndarray
            Softmax probability map of shape (3, H, W).
        stretched : np.ndarray
            Normalized 2D image [0.0, 1.0] after astronomical stretch (leakySB).
        """
        t0 = time.time()

        # Ingest image data
        if isinstance(image, (str, Path)):
            astro_obj = load_astro_image(image)
            raw_data = astro_obj.data
        elif isinstance(image, (FITSImage, ASDFImage)):
            raw_data = image.data
        elif isinstance(image, np.ndarray):
            raw_data = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        h, w = raw_data.shape

        # Astronomical stretch (leakySB)
        stretched = stretch_astro_image(
            raw_data,
            stretch_type=stretch_type,
            zero_points=zero_points,
            pixel_scale=pixel_scale,
            mmin=mmin,
            mmax=mmax,
        )

        # 3-channel input representation (UNet++ expects in_channels=3)
        norm_3ch = np.repeat(stretched[np.newaxis, :, :], 3, axis=0)  # (3, H, W)

        # Case 1: Direct full-frame inference
        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            pad_h = (16 - (h % 16)) % 16
            pad_w = (16 - (w % 16)) % 16

            if pad_h > 0 or pad_w > 0:
                padded = np.pad(norm_3ch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            else:
                padded = norm_3ch

            input_t = torch.from_numpy(padded).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(input_t)
                if isinstance(logits, list):
                    logits = logits[-1]
                probs = F.softmax(logits, dim=1).squeeze(0)[:, :h, :w].cpu().numpy()

            pred_mask = np.argmax(probs, axis=0).astype(np.uint8)
            elapsed = time.time() - t0
            if verbose:
                print(f"[sam2roman] Direct UNet++ inference on {w}x{h} completed in {elapsed:.2f}s on {self.device}.", flush=True)
            return pred_mask, probs, stretched

        # Case 2: Tiled sliding-window inference with cosine window blending
        # Enforce tile size divisible by 16
        tile_size = (tile_size // 16) * 16
        overlap = min(overlap, tile_size // 4)

        def _get_coords(total_len: int, size: int, ov: int) -> List[int]:
            if total_len <= size:
                return [0]
            stride = size - ov
            coords = []
            curr = 0
            while curr + size < total_len:
                coords.append(curr)
                curr += stride
            coords.append(total_len - size)
            return sorted(list(set(coords)))

        y_coords = _get_coords(h, tile_size, overlap)
        x_coords = _get_coords(w, tile_size, overlap)
        num_tiles = len(y_coords) * len(x_coords)

        if verbose:
            print(f"[sam2roman] Starting tiled UNet++ inference: {w}x{h} across {len(x_coords)}x{len(y_coords)} = {num_tiles} tiles "
                  f"(tile_size={tile_size}, overlap={overlap}, batch_size={batch_size}) on {self.device}...", flush=True)

        prob_canvas = np.zeros((3, h, w), dtype=np.float32)
        weight_canvas = np.zeros((h, w), dtype=np.float32)

        # Collect tile coordinates
        tile_list: List[Tuple[int, int]] = []
        for y in y_coords:
            for x in x_coords:
                tile_list.append((y, x))

        total_batches = (len(tile_list) + batch_size - 1) // batch_size

        # Process in batches
        for batch_idx, i in enumerate(range(0, len(tile_list), batch_size)):
            batch_coords = tile_list[i : i + batch_size]
            batch_crops = [norm_3ch[:, y : y + tile_size, x : x + tile_size] for y, x in batch_coords]
            batch_t = torch.from_numpy(np.stack(batch_crops)).to(self.device)

            with torch.no_grad():
                logits = self.model(batch_t)
                if isinstance(logits, list):
                    logits = logits[-1]
                batch_probs = F.softmax(logits, dim=1).cpu().numpy()

            for b_idx, (y, x) in enumerate(batch_coords):
                is_top = (y == 0)
                is_bottom = (y == y_coords[-1])
                is_left = (x == 0)
                is_right = (x == x_coords[-1])
                w2d = self._build_tile_window(tile_size, overlap, is_top, is_bottom, is_left, is_right)

                prob_canvas[:, y : y + tile_size, x : x + tile_size] += batch_probs[b_idx] * w2d
                weight_canvas[y : y + tile_size, x : x + tile_size] += w2d

            del batch_t, logits
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()

            if verbose and ((batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches):
                print(f"[sam2roman]   -> Progress: batch {batch_idx + 1}/{total_batches} "
                      f"({min(i + batch_size, len(tile_list))}/{num_tiles} tiles processed)", flush=True)

        probs = prob_canvas / np.maximum(weight_canvas, 1e-7)
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)

        elapsed = time.time() - t0
        if verbose:
            print(f"[sam2roman] Tiled UNet++ inference completed in {elapsed:.2f}s "
                  f"({num_tiles} tiles, {elapsed/num_tiles*1000:.1f}ms/tile).", flush=True)

        return pred_mask, probs, stretched
