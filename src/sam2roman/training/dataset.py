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
    prompt_type : str, default 'box'
        Type of training prompt to generate: 'box', 'point', or 'both'.
    transform : callable, optional
        Optional augmentations (e.g. torchvision transforms).
    """

    def __init__(
        self,
        images: List[Union[str, Path, np.ndarray, FITSImage]],
        masks: List[np.ndarray],
        stretch_type: str = "asinh",
        prompt_type: str = "box",
        transform: Optional[Any] = None,
    ):
        if len(images) != len(masks):
            raise ValueError(f"Number of images ({len(images)}) does not match masks ({len(masks)})")

        self.images = images
        self.masks = masks
        self.stretch_type = stretch_type
        self.prompt_type = prompt_type
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

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

        # If mask is multi-object (>1 unique non-zero values), pick one random object for prompt-based training
        unique_labels = np.unique(raw_mask)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) > 0:
            target_label = np.random.choice(unique_labels)
            target_mask = (raw_mask == target_label).astype(np.float32)
        else:
            target_mask = (raw_mask > 0).astype(np.float32)

        ys, xs = np.where(target_mask > 0)
        if len(xs) > 0:
            x_min, x_max = float(np.min(xs)), float(np.max(xs))
            y_min, y_max = float(np.min(ys)), float(np.max(ys))
            # Random foreground point prompt
            pt_idx = np.random.randint(0, len(xs))
            point = [float(xs[pt_idx]), float(ys[pt_idx])]
            label = 1
        else:
            # Empty fallback
            h, w = raw_data.shape
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
