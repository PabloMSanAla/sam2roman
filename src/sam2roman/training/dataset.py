"""
PyTorch Dataset for astronomical images and segmentation targets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from astropy.io import fits
from scipy.ndimage import center_of_mass

from sam2roman.io.fits import FITSImage, stretch_astro_image

logger = logging.getLogger(__name__)


class AstronomicalDataset(Dataset):
    """
    Dataset for fine-tuning SAM 2 on astronomical images with ground-truth masks.

    Supports:
    1. Paired directories of FITS images and masks (`from_directories`).
    2. Master FITS tiled slicing (`from_single_fits_tiled`).
    3. `samples_per_file=None`: Uses ALL patches possible (one cutout for every detected object).
    4. `samples_per_file=K`: Extracts K random object cutouts per file per epoch.
    5. Point prompts and bounding box prompts with coordinate translation.

    Parameters
    ----------
    images : List[Union[str, Path, np.ndarray, FITSImage]]
        List of astronomical FITS file paths, FITSImage objects, or 2D numpy arrays.
    masks : List[Union[str, Path, np.ndarray]]
        List of 2D binary masks, multi-object segmentation maps, or FITS mask filepaths.
    stretch_type : str, default 'asinh'
        Astronomical stretch for FITS conversion.
    prompt_type : str, default 'point'
        Type of training prompt: 'point', 'box', or 'both'.
    crop_size : Optional[int], default None
        If specified (e.g. 512), crops a window around a target object.
    samples_per_file : Optional[Union[int, str]], default None
        - None or 'all': Use ALL patches possible (one cutout for every detected object across all files).
        - Integer K (e.g. 10, 20): Extract K cutouts per FITS file per epoch.
    box_jitter_pct : float, default 0.05
        Fraction of box size to randomly jitter coordinates to build model robustness.
    transform : callable, optional
        Optional augmentations.
    """

    def __init__(
        self,
        images: List[Union[str, Path, np.ndarray, FITSImage]],
        masks: List[Union[str, Path, np.ndarray]],
        stretch_type: str = "asinh",
        prompt_type: str = "point",
        crop_size: Optional[int] = None,
        samples_per_file: Optional[Union[int, str]] = None,
        box_jitter_pct: float = 0.05,
        transform: Optional[Any] = None,
    ):
        if len(images) != len(masks):
            raise ValueError(f"Number of images ({len(images)}) does not match masks ({len(masks)})")

        self.images = images
        self.masks = masks
        self.stretch_type = stretch_type
        self.prompt_type = prompt_type
        self.crop_size = crop_size
        self.samples_per_file = samples_per_file
        self.box_jitter_pct = box_jitter_pct
        self.transform = transform

        self.indexed_objects: Optional[List[Dict[str, Any]]] = None

        # If samples_per_file is None or 'all', index every object across all files
        if (self.samples_per_file is None or self.samples_per_file == "all") and self.crop_size is not None:
            self._build_all_objects_index()

    def _build_all_objects_index(self) -> None:
        """Scan masks to index every discrete object (centroid and label) for all patches possible."""
        self.indexed_objects = []
        for file_idx, mask_item in enumerate(self.masks):
            raw_mask = self._load_mask_array(file_idx)
            unique_labels = np.unique(raw_mask)
            unique_labels = unique_labels[unique_labels > 0]

            if len(unique_labels) == 0:
                continue

            centroids = center_of_mass(raw_mask > 0, labels=raw_mask, index=unique_labels)
            for lbl, (cy, cx) in zip(unique_labels, centroids):
                self.indexed_objects.append({
                    "file_idx": file_idx,
                    "target_label": int(lbl),
                    "cx": int(cx),
                    "cy": int(cy),
                })

    def __len__(self) -> int:
        if self.indexed_objects is not None:
            return len(self.indexed_objects)
        elif isinstance(self.samples_per_file, int):
            return len(self.images) * self.samples_per_file
        else:
            return len(self.images)

    @classmethod
    def from_directories(
        cls,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        pattern: str = "*.fits",
        crop_size: Optional[int] = 512,
        samples_per_file: Optional[Union[int, str]] = None,
        prompt_type: str = "point",
        stretch_type: str = "asinh",
        max_samples: Optional[int] = None,
    ) -> "AstronomicalDataset":
        """
        Create dataset by matching image files and mask files from two directories.

        Parameters
        ----------
        images_dir : str or Path
            Directory containing original astronomical FITS files.
        masks_dir : str or Path
            Directory containing corresponding mask FITS files with matching filenames.
        pattern : str, default '*.fits'
            File matching pattern (e.g., '*.fits', '*.fits.gz').
        crop_size : Optional[int], default 512
            Crop window size around target objects (recommended for large 4088x4088 images).
        samples_per_file : Optional[Union[int, str]], default None
            - None or 'all': Use ALL patches possible (one cutout for every detected object).
            - Integer K (e.g. 10, 20): Extract K cutouts per file per epoch.
        prompt_type : str, default 'point'
            Type of prompt to sample: 'point' or 'box'.
        stretch_type : str, default 'asinh'
            Intensity stretch for astronomical conversion.
        max_samples : Optional[int], default None
            Maximum number of FITS files to match.
        """
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)

        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")
        if not masks_path.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_path}")

        img_files = sorted(list(images_path.glob(pattern)))
        matched_images = []
        matched_masks = []

        for img_p in img_files:
            mask_p = masks_path / img_p.name
            if not mask_p.exists():
                candidates = list(masks_path.glob(f"{img_p.stem}.*"))
                if candidates:
                    mask_p = candidates[0]
                else:
                    continue

            matched_images.append(img_p)
            matched_masks.append(mask_p)

            if max_samples is not None and len(matched_images) >= max_samples:
                break

        if not matched_images:
            raise ValueError(
                f"No matching file pairs found between '{images_path}' and '{masks_path}' with pattern '{pattern}'"
            )

        return cls(
            images=matched_images,
            masks=matched_masks,
            stretch_type=stretch_type,
            prompt_type=prompt_type,
            crop_size=crop_size,
            samples_per_file=samples_per_file,
        )

    @classmethod
    def from_single_fits_tiled(
        cls,
        fits_image: Union[FITSImage, np.ndarray, str, Path],
        segmap: Union[np.ndarray, str, Path],
        tile_size: int = 512,
        stride: int = 384,
        min_objects: int = 1,
        stretch_type: str = "asinh",
        prompt_type: str = "point",
    ) -> "AstronomicalDataset":
        """Create a multi-sample dataset from ONE single large FITS image and segmap by slicing into tiles."""
        if isinstance(fits_image, (str, Path)):
            raw_data = FITSImage.from_file(fits_image).data
        elif isinstance(fits_image, FITSImage):
            raw_data = fits_image.data
        else:
            raw_data = np.asarray(fits_image, dtype=np.float32)

        if isinstance(segmap, (str, Path)):
            with fits.open(segmap) as hdul:
                hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                seg_data = np.squeeze(hdu.data).astype(np.int32)
        else:
            seg_data = np.asarray(segmap, dtype=np.int32)

        h, w = raw_data.shape
        tiles_img = []
        tiles_mask = []

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                patch_img = raw_data[y : y + tile_size, x : x + tile_size]
                patch_mask = seg_data[y : y + tile_size, x : x + tile_size]

                unique_objs = np.unique(patch_mask)
                unique_objs = unique_objs[unique_objs > 0]
                if len(unique_objs) >= min_objects:
                    tiles_img.append(patch_img)
                    tiles_mask.append(patch_mask)

        if not tiles_img:
            cy, cx = h // 2, w // 2
            tiles_img.append(raw_data[cy - tile_size // 2 : cy + tile_size // 2, cx - tile_size // 2 : cx + tile_size // 2])
            tiles_mask.append(seg_data[cy - tile_size // 2 : cy + tile_size // 2, cx - tile_size // 2 : cx + tile_size // 2])

        return cls(
            images=tiles_img,
            masks=tiles_mask,
            stretch_type=stretch_type,
            prompt_type=prompt_type,
        )

    def _load_image_array(self, idx: int) -> np.ndarray:
        item = self.images[idx]
        if isinstance(item, (str, Path)):
            with fits.open(item, memmap=True) as hdul:
                hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                return np.squeeze(hdu.data).astype(np.float32)
        elif isinstance(item, FITSImage):
            return item.data
        elif isinstance(item, np.ndarray):
            return item.astype(np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(item)}")

    def _load_mask_array(self, idx: int) -> np.ndarray:
        item = self.masks[idx]
        if isinstance(item, (str, Path)):
            with fits.open(item, memmap=True) as hdul:
                hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                return np.squeeze(hdu.data).astype(np.int32)
        return np.asarray(item, dtype=np.int32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.indexed_objects is not None:
            # Mode: ALL patches possible (exact object index)
            obj_meta = self.indexed_objects[idx]
            file_idx = obj_meta["file_idx"]
            target_label = obj_meta["target_label"]
            cx, cy = obj_meta["cx"], obj_meta["cy"]
        elif isinstance(self.samples_per_file, int):
            # Mode: K samples per file
            file_idx = idx // self.samples_per_file
            target_label = None
            cx, cy = None, None
        else:
            file_idx = idx
            target_label = None
            cx, cy = None, None

        raw_data = self._load_image_array(file_idx)
        raw_mask = self._load_mask_array(file_idx)
        h, w = raw_data.shape

        # Select target object if not pre-indexed
        if target_label is None:
            unique_labels = np.unique(raw_mask)
            unique_labels = unique_labels[unique_labels > 0]
            if len(unique_labels) > 0:
                target_label = int(np.random.choice(unique_labels))
                ys_all, xs_all = np.where(raw_mask == target_label)
                cx, cy = int(np.mean(xs_all)), int(np.mean(ys_all))
            else:
                target_label = 0
                cx, cy = w // 2, h // 2

        # Cutout extraction around target object
        if self.crop_size is not None and (h > self.crop_size or w > self.crop_size):
            cs = self.crop_size
            x0 = max(0, min(w - cs, cx - cs // 2 + np.random.randint(-15, 16)))
            y0 = max(0, min(h - cs, cy - cs // 2 + np.random.randint(-15, 16)))

            data_patch = raw_data[y0 : y0 + cs, x0 : x0 + cs]
            mask_patch = (raw_mask[y0 : y0 + cs, x0 : x0 + cs] == target_label).astype(np.float32)

            ys, xs = np.where(mask_patch > 0)
            if len(xs) > 0:
                pt_idx = np.random.randint(0, len(xs))
                point = [float(xs[pt_idx]), float(ys[pt_idx])]
                label = 1
                x_min, x_max = float(np.min(xs)), float(np.max(xs))
                y_min, y_max = float(np.min(ys)), float(np.max(ys))
            else:
                point = [float(cs // 2), float(cs // 2)]
                label = 0
                x_min, y_min, x_max, y_max = 0.0, 0.0, float(cs), float(cs)
        else:
            data_patch = raw_data
            mask_patch = (raw_mask == target_label).astype(np.float32) if target_label > 0 else (raw_mask > 0).astype(np.float32)

            ys, xs = np.where(mask_patch > 0)
            if len(xs) > 0:
                pt_idx = np.random.randint(0, len(xs))
                point = [float(xs[pt_idx]), float(ys[pt_idx])]
                label = 1
                x_min, x_max = float(np.min(xs)), float(np.max(xs))
                y_min, y_max = float(np.min(ys)), float(np.max(ys))
            else:
                point = [float(w // 2), float(h // 2)]
                label = 0
                x_min, y_min, x_max, y_max = 0.0, 0.0, float(w), float(h)

        norm = stretch_astro_image(data_patch, stretch_type=self.stretch_type)
        rgb_uint8 = (norm * 255.0).astype(np.uint8)
        rgb_3ch = np.stack([rgb_uint8, rgb_uint8, rgb_uint8], axis=-1)
        pil_image = Image.fromarray(rgb_3ch, mode="RGB")

        bw = max(x_max - x_min, 1.0)
        bh = max(y_max - y_min, 1.0)
        if self.box_jitter_pct > 0.0:
            jx = (np.random.rand() * 2 - 1) * bw * self.box_jitter_pct
            jy = (np.random.rand() * 2 - 1) * bh * self.box_jitter_pct
            x_min = max(0.0, x_min + jx)
            x_max = min(float(data_patch.shape[1]), x_max + jx)
            y_min = max(0.0, y_min + jy)
            y_max = min(float(data_patch.shape[0]), y_max + jy)

        box = [x_min, y_min, x_max, y_max]

        image_tensor = torch.from_numpy(rgb_3ch).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)
        box_tensor = torch.tensor(box, dtype=torch.float32)
        point_tensor = torch.tensor([point], dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.int64)

        return {
            "image": image_tensor,
            "pil_image": pil_image,
            "mask": mask_tensor,
            "box": box_tensor,
            "point": point_tensor,
            "point_label": label_tensor,
            "raw_data": data_patch,
        }
