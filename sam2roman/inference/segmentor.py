"""
SAM 2 astronomical segmentation engine: high-performance automatic mask generation,
prompt inference, and decoupled astronomical photometry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam2roman.io.fits import FITSImage
from sam2roman.models.sam2 import SAM2ModelWrapper, DEFAULT_SAM2_MODEL
from sam2roman.utils.metrics import compute_source_photometry

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """
    Container for astronomical segmentation outputs.
    """
    masks: List[np.ndarray] = field(default_factory=list)
    boxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    areas: List[int] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    sky_centroids: Optional[List[Tuple[float, float]]] = None
    fluxes: Optional[List[float]] = None
    photometry: Optional[List[Dict[str, Any]]] = None
    image_shape: Tuple[int, int] = (0, 0)

    def __len__(self) -> int:
        return len(self.masks)

    def filter_by_area(self, min_area: int = 10, max_area: Optional[int] = None) -> "SegmentationResult":
        """Filter out masks that are too small (noise/cosmic rays) or too large (sky background)."""
        kept_indices = []
        for i, a in enumerate(self.areas):
            if a >= min_area and (max_area is None or a <= max_area):
                kept_indices.append(i)

        return self._subset(kept_indices)

    def filter_by_score(self, min_score: float = 0.5) -> "SegmentationResult":
        """Filter out masks below predicted IoU / confidence score."""
        kept_indices = [i for i, s in enumerate(self.scores) if s >= min_score]
        return self._subset(kept_indices)

    def _subset(self, indices: List[int]) -> "SegmentationResult":
        sub_masks = [self.masks[i] for i in indices]
        sub_boxes = [self.boxes[i] for i in indices]
        sub_scores = [self.scores[i] for i in indices]
        sub_areas = [self.areas[i] for i in indices]
        sub_centroids = [self.centroids[i] for i in indices]
        sub_sky = [self.sky_centroids[i] for i in indices] if self.sky_centroids is not None else None
        sub_fluxes = [self.fluxes[i] for i in indices] if self.fluxes is not None else None
        sub_phot = [self.photometry[i] for i in indices] if self.photometry is not None else None

        return SegmentationResult(
            masks=sub_masks,
            boxes=sub_boxes,
            scores=sub_scores,
            areas=sub_areas,
            centroids=sub_centroids,
            sky_centroids=sub_sky,
            fluxes=sub_fluxes,
            photometry=sub_phot,
            image_shape=self.image_shape,
        )

    def to_combined_mask(self) -> np.ndarray:
        """
        Merge individual boolean masks into a single 2D integer labeled array.
        0 = background, 1..N = object ID.
        """
        if not self.masks:
            return np.zeros(self.image_shape, dtype=np.int32)

        combined = np.zeros(self.image_shape, dtype=np.int32)
        # Sort by area descending so smaller objects are painted on top of larger ones
        sorted_indices = np.argsort(self.areas)[::-1]
        for rank, idx in enumerate(sorted_indices, start=1):
            combined[self.masks[idx]] = idx + 1
        return combined

    def compute_photometry(
        self,
        image: Union[FITSImage, np.ndarray, str, Path],
        bkg_median: Optional[float] = None,
        bkg_std: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Measure astronomical source photometry on the detected segmentation masks.

        Parameters
        ----------
        image : FITSImage, np.ndarray, or Path
            Raw astronomical image with un-stretched electron/flux counts.
        """
        t0 = time.time()
        if isinstance(image, (str, Path)):
            raw_data = FITSImage.from_file(image).data
        elif isinstance(image, FITSImage):
            raw_data = image.data
        elif isinstance(image, np.ndarray):
            raw_data = image
        else:
            raise TypeError(f"Unsupported image type for photometry: {type(image)}")

        catalog = compute_source_photometry(
            data=raw_data,
            masks=self.masks,
            bkg_median=bkg_median,
            bkg_std=bkg_std,
        )

        self.photometry = catalog
        self.fluxes = [item["net_flux"] for item in catalog]

        if verbose:
            print(
                f"[sam2roman] Photometry computed for {len(catalog)} sources in {time.time() - t0:.2f}s."
            )

        return catalog


class SAM2Segmentor:
    """
    High-performance segmentor for running SAM 2 inference on astronomical images.

    Optimized for large astronomical detectors (e.g. Roman WFI 4088x4088) with:
    1. Single-pass image feature encoding (Hiera vision backbone runs once).
    2. Batched prompt query decoding via cached embeddings.
    3. Low-resolution box-gated Non-Maximum Suppression (zero 17GB memory blowup).
    4. Fast astronomical source peak candidate detection mode.
    5. Decoupled source photometry step.
    """

    def __init__(
        self,
        model_wrapper: Optional[SAM2ModelWrapper] = None,
        model_id: str = DEFAULT_SAM2_MODEL,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if model_wrapper is not None:
            self.model_wrapper = model_wrapper
        elif checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt_model_id = ckpt.get("model_id", model_id)
            self.model_wrapper = SAM2ModelWrapper(model_id=ckpt_model_id, device=device)
            self.model_wrapper.model.load_state_dict(ckpt["model_state_dict"])
            self.model_wrapper.model.eval()
            logger.info(f"Loaded fine-tuned checkpoint from '{ckpt_path}' (model_id: {ckpt_model_id}) on {self.model_wrapper.device}")
        else:
            self.model_wrapper = SAM2ModelWrapper(model_id=model_id, device=device)

        if checkpoint_path is not None and model_wrapper is not None:
            self.load_checkpoint(checkpoint_path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        model_id: Optional[str] = None,
    ) -> "SAM2Segmentor":
        """
        Instantiate SAM2Segmentor directly from a fine-tuned training checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the saved checkpoint file (e.g. 'checkpoints/checkpoint_best.pt').
        device : str or torch.device, optional
            Target accelerator ('cuda', 'mps', 'cpu'). Auto-detected if None.
        model_id : str, optional
            Base model ID. If None, auto-detected from checkpoint metadata.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        target_model_id = model_id or ckpt.get("model_id", DEFAULT_SAM2_MODEL)

        wrapper = SAM2ModelWrapper(model_id=target_model_id, device=device)
        wrapper.model.load_state_dict(ckpt["model_state_dict"])
        wrapper.model.eval()

        logger.info(
            f"Initialized fine-tuned SAM2Segmentor from '{ckpt_path.name}' (model: {target_model_id}) on {wrapper.device}"
        )
        return cls(model_wrapper=wrapper)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load fine-tuned model state_dict from a checkpoint into the active segmentor.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint file.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.model_wrapper.device)
        self.model_wrapper.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_wrapper.model.eval()
        logger.info(f"Loaded fine-tuned checkpoint into SAM2Segmentor: {ckpt_path}")

    def _prepare_inputs(
        self,
        image: Union[FITSImage, np.ndarray, Image.Image, str, Path],
        stretch_type: str = "leakysb",
    ) -> Tuple[Image.Image, Optional[FITSImage], Optional[np.ndarray]]:
        """
        Convert image input into a PIL RGB Image, extracting underlying FITS data and WCS.
        """
        fits_obj: Optional[FITSImage] = None
        raw_data: Optional[np.ndarray] = None

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.suffix.lower() in (".fits", ".fit", ".fts", ".gz"):
                fits_obj = FITSImage.from_file(path)
                pil_img = fits_obj.to_pil(stretch_type=stretch_type)
                raw_data = fits_obj.data
            else:
                pil_img = Image.open(path).convert("RGB")
        elif isinstance(image, FITSImage):
            fits_obj = image
            pil_img = fits_obj.to_pil(stretch_type=stretch_type)
            raw_data = fits_obj.data
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                fits_obj = FITSImage(image)
                pil_img = fits_obj.to_pil(stretch_type=stretch_type)
                raw_data = image
            elif image.ndim == 3 and image.shape[2] in (1, 3):
                if image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=-1)
                if image.dtype != np.uint8:
                    norm = (image - np.min(image)) / (np.ptp(image) + 1e-6)
                    image = (norm * 255).astype(np.uint8)
                pil_img = Image.fromarray(image, mode="RGB")
            else:
                raise ValueError(f"Unsupported array shape: {image.shape}")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return pil_img, fits_obj, raw_data

    def segment_everything(
        self,
        image: Union[FITSImage, np.ndarray, Image.Image, str, Path],
        prompt_mode: str = "grid",
        points_per_side: int = 32,
        max_astro_candidates: int = 400,
        batch_size: int = 64,
        pred_iou_thresh: float = 0.65,
        min_mask_region_area: int = 15,
        max_mask_region_area: Optional[int] = None,
        iou_overlap_thresh: float = 0.75,
        stretch_type: str = "leakysb",
        verbose: bool = True,
    ) -> SegmentationResult:
        """
        Segment all astronomical sources in the FITS image.

        High-performance single-pass implementation:
        Encodes the image once, evaluates candidate points directly against cached embeddings,
        and deduplicates at low resolution before high-resolution upsampling.

        Parameters
        ----------
        image : FITSImage, np.ndarray, PIL Image, or file path
            The input astronomical image.
        prompt_mode : str, default 'grid'
            - 'grid': Uniform N x N point grid across the image.
            - 'astronomical': Fast 2D peak candidate detection (extremely fast on large empty fields).
        points_per_side : int, default 32
            Used when prompt_mode='grid'. 32 yields 1,024 prompt queries.
        max_astro_candidates : int, default 400
            Max candidate peaks when prompt_mode='astronomical'.
        batch_size : int, default 64
            Number of prompt points evaluated per forward pass of the mask decoder.
        pred_iou_thresh : float, default 0.65
            Predicted IoU threshold to reject low-confidence masks.
        min_mask_region_area : int, default 15
            Minimum pixel count for keeping an astronomical mask (filters cosmic rays/hot pixels).
        max_mask_region_area : int, optional
            Maximum pixel count (filters full-detector background masks).
        iou_overlap_thresh : float, default 0.75
            IoU threshold for deduplicating overlapping masks.
        stretch_type : str, default 'leakysb'
            Intensity stretch for FITS conversion ('leakysb', 'asinh', 'zscale', 'linear', 'log').
        verbose : bool, default True
            Whether to print detailed timing and progress diagnostics.

        Returns
        -------
        SegmentationResult
            Container with extracted masks, bounding boxes, scores, centroids, and WCS coordinates.
            (Photometry is separate via results.compute_photometry(image)).
        """
        t_total_start = time.time()

        # Step 1: Preprocessing & Stretch
        t0 = time.time()
        pil_img, fits_obj, raw_data = self._prepare_inputs(image, stretch_type=stretch_type)
        width, height = pil_img.size
        t_prep = time.time() - t0

        if verbose:
            print(f"[sam2roman] Preprocessed image ({width} x {height}, {stretch_type} stretch) in {t_prep:.2f}s")

        model = self.model_wrapper.model
        processor = self.model_wrapper.processor
        device = self.model_wrapper.device

        # Step 2: Vision Backbone Run ONCE
        t0 = time.time()
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(inputs.pixel_values)
        t_embed = time.time() - t0

        if verbose:
            print(f"[sam2roman] Vision backbone embedding cached in {t_embed:.2f}s on {device}")

        # Step 3: Generate Prompt Points
        t0 = time.time()
        if prompt_mode == "astronomical" and raw_data is not None:
            points = self._detect_astronomical_peaks(raw_data, max_peaks=max_astro_candidates)
            if len(points) == 0:
                # Fallback to grid if no peaks detected
                points = self._generate_point_grid(width, height, points_per_side)
        else:
            points = self._generate_point_grid(width, height, points_per_side)
        t_pts = time.time() - t0

        if verbose:
            print(f"[sam2roman] Generated {len(points)} prompt points (mode='{prompt_mode}') in {t_pts:.3f}s")

        # Step 4: Batched Prompt Decoding via Cached Embeddings
        t0 = time.time()
        candidate_masks_lowres: List[torch.Tensor] = []
        candidate_boxes_lowres: List[List[float]] = []
        candidate_scores: List[float] = []

        for start_idx in range(0, len(points), batch_size):
            b_pts = points[start_idx : start_idx + batch_size]
            pts_nested = [ [[float(p[0]), float(p[1])]] for p in b_pts ]
            lbls_nested = [ [1] for _ in b_pts ]

            proc_batch = processor(
                images=pil_img,
                input_points=[pts_nested],
                input_labels=[lbls_nested],
                return_tensors="pt",
            )
            pts_tensor = proc_batch.input_points.to(device)
            lbls_tensor = proc_batch.input_labels.to(device)

            with torch.no_grad():
                out = model(
                    image_embeddings=image_embeddings,
                    input_points=pts_tensor,
                    input_labels=lbls_tensor,
                )

            # out.pred_masks is (1, B, 3, 256, 256)
            # out.iou_scores is (1, B, 3)
            best_mask_indices = torch.argmax(out.iou_scores[0], dim=-1)  # (B,)

            for b in range(len(b_pts)):
                score = float(out.iou_scores[0, b, best_mask_indices[b]])
                if score >= pred_iou_thresh:
                    mask_low = out.pred_masks[0, b, best_mask_indices[b]]  # (256, 256)
                    mask_bool = mask_low > 0.0
                    if mask_bool.any():
                        ys, xs = torch.where(mask_bool)
                        x0, y0 = float(xs.min()), float(ys.min())
                        x1, y1 = float(xs.max()), float(ys.max())
                        candidate_masks_lowres.append(mask_low.unsqueeze(0))  # (1, 256, 256)
                        candidate_boxes_lowres.append([x0, y0, x1, y1])
                        candidate_scores.append(score)

        t_decode = time.time() - t0
        if verbose:
            print(
                f"[sam2roman] Evaluated {len(points)} prompts in {t_decode:.2f}s "
                f"({len(candidate_scores)} candidate masks pass IoU threshold >= {pred_iou_thresh})"
            )

        # Step 5: Fast Low-Resolution Box-Gated Deduplication (NMS)
        t0 = time.time()
        kept_indices = self._fast_box_gated_dedup(
            candidate_masks_lowres,
            candidate_boxes_lowres,
            candidate_scores,
            iou_thresh=iou_overlap_thresh,
        )
        t_dedup = time.time() - t0

        if verbose:
            print(f"[sam2roman] Deduplicated to {len(kept_indices)} unique objects in {t_dedup:.3f}s")

        if not kept_indices:
            return SegmentationResult(image_shape=(height, width))

        # Step 6: Upsample Only the Kept Masks to Full Resolution
        t0 = time.time()
        kept_tensors = torch.stack([candidate_masks_lowres[idx] for idx in kept_indices], dim=0)  # (K, 1, 256, 256)
        upsampled_logits = F.interpolate(
            kept_tensors,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (K, H, W)

        upsampled_masks = (upsampled_logits > 0.0).cpu().numpy()
        t_upsample = time.time() - t0

        # Build clean masks, boxes, areas, centroids
        masks: List[np.ndarray] = []
        boxes: List[List[float]] = []
        scores: List[float] = []
        areas: List[int] = []
        centroids: List[Tuple[float, float]] = []
        sky_centroids: Optional[List[Tuple[float, float]]] = [] if (fits_obj and fits_obj.wcs) else None

        for k, idx in enumerate(kept_indices):
            m_bool = upsampled_masks[k]
            area = int(np.sum(m_bool))

            if area < min_mask_region_area:
                continue
            if max_mask_region_area is not None and area > max_mask_region_area:
                continue

            ys, xs = np.where(m_bool)
            if len(xs) == 0:
                continue

            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            box = [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]

            masks.append(m_bool)
            boxes.append(box)
            scores.append(candidate_scores[idx])
            areas.append(area)
            centroids.append((cx, cy))

            if fits_obj and fits_obj.wcs and sky_centroids is not None:
                sky = fits_obj.pixel_to_sky(cx, cy)
                if sky:
                    sky_centroids.append(sky)

        t_total = time.time() - t_total_start
        if verbose:
            print(
                f"[sam2roman] Segmentation complete: {len(masks)} sources extracted in {t_total:.2f}s total."
            )

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            areas=areas,
            centroids=centroids,
            sky_centroids=sky_centroids,
            image_shape=(height, width),
        )

    def compute_photometry(
        self,
        image: Union[FITSImage, np.ndarray, str, Path],
        results: SegmentationResult,
        bkg_median: Optional[float] = None,
        bkg_std: Optional[float] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Measure astronomical source photometry for a SegmentationResult.
        """
        return results.compute_photometry(
            image=image,
            bkg_median=bkg_median,
            bkg_std=bkg_std,
            verbose=verbose,
        )

    def segment_points(
        self,
        image: Union[FITSImage, np.ndarray, Image.Image, str, Path],
        points: List[Tuple[float, float]],
        labels: Optional[List[int]] = None,
        stretch_type: str = "leakysb",
        verbose: bool = False,
    ) -> SegmentationResult:
        """
        Segment sources using targeted point prompts (e.g. from an astronomical catalog).
        """
        t0 = time.time()
        pil_img, fits_obj, _ = self._prepare_inputs(image, stretch_type=stretch_type)
        width, height = pil_img.size

        if labels is None:
            labels = [1] * len(points)

        model = self.model_wrapper.model
        processor = self.model_wrapper.processor
        device = self.model_wrapper.device

        pts_nested = [ [[float(p[0]), float(p[1])]] for p in points ]
        lbls_nested = [ [int(l)] for l in labels ]

        proc_inputs = processor(
            images=pil_img,
            input_points=[pts_nested],
            input_labels=[lbls_nested],
            return_tensors="pt",
        )
        pts_tensor = proc_inputs.input_points.to(device)
        lbls_tensor = proc_inputs.input_labels.to(device)

        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(inputs.pixel_values)
            out = model(
                image_embeddings=image_embeddings,
                input_points=pts_tensor,
                input_labels=lbls_tensor,
            )

        best_mask_indices = torch.argmax(out.iou_scores[0], dim=-1)  # (N,)
        kept_masks_low = [out.pred_masks[0, i, best_mask_indices[i]].unsqueeze(0) for i in range(len(points))]
        kept_tensors = torch.stack(kept_masks_low, dim=0)

        upsampled_logits = F.interpolate(
            kept_tensors,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        upsampled_masks = (upsampled_logits > 0.0).cpu().numpy()

        masks: List[np.ndarray] = []
        boxes: List[List[float]] = []
        scores: List[float] = []
        areas: List[int] = []
        centroids: List[Tuple[float, float]] = []

        for i in range(len(points)):
            m_bool = upsampled_masks[i]
            score = float(out.iou_scores[0, i, best_mask_indices[i]])
            ys, xs = np.where(m_bool)
            if len(xs) > 0:
                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                box = [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]
            else:
                cx, cy = float(points[i][0]), float(points[i][1])
                box = [cx, cy, cx, cy]

            masks.append(m_bool)
            boxes.append(box)
            scores.append(score)
            areas.append(int(np.sum(m_bool)))
            centroids.append((cx, cy))

        if verbose:
            print(f"[sam2roman] Point prompt segmentation completed for {len(points)} points in {time.time() - t0:.2f}s")

        return SegmentationResult(
            masks=masks,
            boxes=boxes,
            scores=scores,
            areas=areas,
            centroids=centroids,
            image_shape=(height, width),
        )

    @staticmethod
    def _generate_point_grid(width: int, height: int, points_per_side: int) -> np.ndarray:
        """Generate regular N x N grid of coordinates across image bounds."""
        xs = np.linspace(width * 0.05, width * 0.95, points_per_side)
        ys = np.linspace(height * 0.05, height * 0.95, points_per_side)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    @staticmethod
    def _detect_astronomical_peaks(data: np.ndarray, max_peaks: int = 400) -> np.ndarray:
        """
        Fast astronomical peak candidate detection.
        Downsamples 2x if image is huge, thresholds at 3-sigma, finds local peaks.
        """
        from scipy.ndimage import maximum_filter

        h, w = data.shape
        step = 2 if (h > 2000 or w > 2000) else 1
        sub_data = data[::step, ::step]

        # Fast background estimate
        med = float(np.median(sub_data[::4, ::4]))
        std = float(np.std(sub_data[::4, ::4]))
        thresh = med + 3.0 * std

        max_filt = maximum_filter(sub_data, size=5)
        peaks = (sub_data > thresh) & (sub_data == max_filt)
        ys, xs = np.where(peaks)

        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)

        peak_vals = sub_data[ys, xs]
        # Sort descending by peak brightness
        sort_order = np.argsort(peak_vals)[::-1][:max_peaks]

        x_coords = xs[sort_order].astype(np.float32) * step
        y_coords = ys[sort_order].astype(np.float32) * step

        return np.stack([x_coords, y_coords], axis=1)

    @staticmethod
    def _fast_box_gated_dedup(
        masks_lowres: List[torch.Tensor],
        boxes_lowres: List[List[float]],
        scores: List[float],
        iou_thresh: float = 0.75,
    ) -> List[int]:
        """
        Fast box-gated deduplication on low-resolution masks (256x256).
        Checks bounding box intersection first; only computes mask IoU if boxes overlap.
        """
        if not scores:
            return []

        order = np.argsort(scores)[::-1]
        keep: List[int] = []

        for i in order:
            bx0, by0, bx1, by1 = boxes_lowres[i]
            m_i = masks_lowres[i] > 0.0
            overlap = False

            for k in keep:
                kx0, ky0, kx1, ky1 = boxes_lowres[k]
                # Fast bbox disjoint test
                if bx0 > kx1 or bx1 < kx0 or by0 > ky1 or by1 < ky0:
                    continue

                m_k = masks_lowres[k] > 0.0
                intersection = (m_i & m_k).sum().item()
                union = (m_i | m_k).sum().item()
                if union > 0 and (intersection / union) > iou_thresh:
                    overlap = True
                    break

            if not overlap:
                keep.append(int(i))

        return keep
