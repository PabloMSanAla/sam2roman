"""
PyTorch Dataset for astronomical images and segmentation targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from sam2roman.io.fits import FITSImage, stretch_astro_image


class AstronomicalDataset(Dataset):
    """
    Dataset for fine-tuning SAM 2 on astronomical images with ground-truth masks.

    Automatically extracts bounding boxes or positive point prompts from masks
    to feed into SAM 2's prompt encoder during training.

    Parameters
    ----------
    images : List[Union[str, Path, np.ndarray, FITSImage]]
        List of astronomical FITS file paths, FITSImage objects, or 2D numpy arrays.
    masks : List[np.ndarray]
        List of 2D binary masks or multi-object segmentation maps (integer labeled).
    stretch_type : str, default 'asinh'
        Astronomical stretch for FITS conversion.
    prompt_type : str, default 'point'
        Type of training prompt: 'point', 'box', or 'both'.
    box_jitter_pct : float, default 0.05
        Fraction of box size to randomly jitter coordinates to build model robustness.
    transform : callable, optional
        Optional augmentations.
    """

    def __init__(
        self,
        images: List[Union[str, Path, np.ndarray, FITSImage]],
        masks: List[np.ndarray],
        stretch_type: str = "asinh",
        prompt_type: str = "point",
        box_jitter_pct: float = 0.05,
        transform: Optional[Any] = None,
    ):
        if len(images) != len(masks):
            raise ValueError(f"Number of images ({len(images)}) does not match masks ({len(masks)})")

        self.images = images
        self.masks = masks
        self.stretch_type = stretch_type
        self.prompt_type = prompt_type
        self.box_jitter_pct = box_jitter_pct
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    @classmethod
    def from_single_fits_tiled(
        cls,
        fits_image: Union[FITSImage, np.ndarray, str, Path],
        segmap: np.ndarray,
        tile_size: int = 512,
        stride: int = 384,
        min_objects: int = 1,
        stretch_type: str = "asinh",
        prompt_type: str = "point",
    ) -> "AstronomicalDataset":
        """
        Create a multi-sample dataset from ONE single large astronomical FITS image and segmap
        by slicing it into overlapping tiles/cutouts (e.g. 512x512).

        Parameters
        ----------
        fits_image : FITSImage, np.ndarray, or Path
            The single master FITS image (e.g. 4088x4088 Roman WFI image).
        segmap : np.ndarray
            Matching 2D integer labeled segmentation map (0=bg, >0=object_id).
        tile_size : int, default 512
            Width and height of each training cutout.
        stride : int, default 384
            Step size between cutouts.
        min_objects : int, default 1
            Minimum number of discrete objects required to retain a tile.
        """
        if isinstance(fits_image, (str, Path)):
            raw_data = FITSImage.from_file(fits_image).data
        elif isinstance(fits_image, FITSImage):
            raw_data = fits_image.data
        else:
            raw_data = np.asarray(fits_image, dtype=np.float32)

        h, w = raw_data.shape
        tiles_img = []
        tiles_mask = []

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                patch_img = raw_data[y : y + tile_size, x : x + tile_size]
                patch_mask = segmap[y : y + tile_size, x : x + tile_size]

                unique_objs = np.unique(patch_mask)
                unique_objs = unique_objs[unique_objs > 0]
                if len(unique_objs) >= min_objects:
                    tiles_img.append(patch_img)
                    tiles_mask.append(patch_mask)

        # Fallback if no tiles passed filter: take center crop
        if not tiles_img:
            cy, cx = h // 2, w // 2
            tiles_img.append(raw_data[cy - tile_size // 2 : cy + tile_size // 2, cx - tile_size // 2 : cx + tile_size // 2])
            tiles_mask.append(segmap[cy - tile_size // 2 : cy + tile_size // 2, cx - tile_size // 2 : cx + tile_size // 2])

        return cls(
            images=tiles_img,
            masks=tiles_mask,
            stretch_type=stretch_type,
            prompt_type=prompt_type,
        )

    def _load_image_array(self, idx: int) -> np.ndarray:
        item = self.images[idx]
        if isinstance(item, (str, Path)):
            fits_obj = FITSImage.from_file(item)
            return fits_obj.data
        elif isinstance(item, FITSImage):
            return item.data
        elif isinstance(item, np.ndarray):
            return item.astype(np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(item)}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_data = self._load_image_array(idx)
        raw_mask = np.asarray(self.masks[idx], dtype=np.int32)

        # Stretch to RGB uint8
        norm = stretch_astro_image(raw_data, stretch_type=self.stretch_type)
        rgb_uint8 = (norm * 255.0).astype(np.uint8)
        rgb_3ch = np.stack([rgb_uint8, rgb_uint8, rgb_uint8], axis=-1)
        pil_image = Image.fromarray(rgb_3ch, mode="RGB")

        # Pick one object from the mask for prompt-target training
        unique_labels = np.unique(raw_mask)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) > 0:
            target_label = np.random.choice(unique_labels)
            target_mask = (raw_mask == target_label).astype(np.float32)
        else:
            target_mask = (raw_mask > 0).astype(np.float32)

        ys, xs = np.where(target_mask > 0)
        h, w = raw_data.shape

        if len(xs) > 0:
            x_min, x_max = float(np.min(xs)), float(np.max(xs))
            y_min, y_max = float(np.min(ys)), float(np.max(ys))

            # Sample point prompt from object foreground
            pt_idx = np.random.randint(0, len(xs))
            point = [float(xs[pt_idx]), float(ys[pt_idx])]
            label = 1

            # Bounding box with optional small jitter
            bw = max(x_max - x_min, 1.0)
            bh = max(y_max - y_min, 1.0)
            if self.box_jitter_pct > 0.0:
                jx = (np.random.rand() * 2 - 1) * bw * self.box_jitter_pct
                jy = (np.random.rand() * 2 - 1) * bh * self.box_jitter_pct
                x_min = max(0.0, x_min + jx)
                x_max = min(float(w), x_max + jx)
                y_min = max(0.0, y_min + jy)
                y_max = min(float(h), y_max + jy)
            box = [x_min, y_min, x_max, y_max]
        else:
            # Fallback
            x_min, y_min, x_max, y_max = 0.0, 0.0, float(w), float(h)
            point = [float(w // 2), float(h // 2)]
            label = 0
            box = [x_min, y_min, x_max, y_max]

        # Convert to tensors
        image_tensor = torch.from_numpy(rgb_3ch).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(target_mask).float().unsqueeze(0)  # (1, H, W)
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
            "raw_data": raw_data,
        }
