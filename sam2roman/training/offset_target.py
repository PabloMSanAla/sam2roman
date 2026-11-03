"""
Target generation utilities for Center Offset Voting instance segmentation.
=============================================================================

Generates multi-task training targets from 2D integer instance segmentation masks:
1. Binary foreground semantic mask (H, W).
2. Gaussian center probability heatmap (H, W) centered at each source centroid.
3. Dense 2D offset vector field (2, H, W) pointing from each pixel to its instance centroid:
   - Channel 0: delta_y = cy - y
   - Channel 1: delta_x = cx - x
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def compute_instance_centroids(
    instance_mask: np.ndarray,
    min_area: int = 1,
) -> Dict[int, Tuple[float, float, int]]:
    """
    Extract geometric center of mass (cy, cx) and pixel area for each instance.

    Parameters
    ----------
    instance_mask : np.ndarray
        2D integer array (H, W) where 0 is background and k > 0 are instance IDs.
    min_area : int, default 1
        Minimum pixel count to retain an instance.

    Returns
    -------
    Dict[int, Tuple[float, float, int]]:
        Mapping from instance_id to (cy, cx, area).
    """
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    centroids: Dict[int, Tuple[float, float, int]] = {}

    for inst_id in unique_ids:
        ys, xs = np.where(instance_mask == inst_id)
        area = len(ys)
        if area < min_area:
            continue
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        centroids[int(inst_id)] = (cy, cx, area)

    return centroids


def generate_center_offset_targets_numpy(
    instance_mask: np.ndarray,
    sigma: float = 2.0,
    min_area: int = 1,
    adaptive_sigma: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Generate foreground mask, Gaussian center heatmap, and (dy, dx) offset field (NumPy).

    Parameters
    ----------
    instance_mask : np.ndarray
        Integer array of shape (H, W) with instance IDs (0 is background).
    sigma : float, default 2.0
        Standard deviation for Gaussian center peaks in pixels.
    min_area : int, default 1
        Minimum pixel count for an instance.
    adaptive_sigma : bool, default False
        If True, scales sigma with the equivalent circular radius of each source.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        - fg_mask: (H, W) float32 in {0.0, 1.0}
        - center_map: (H, W) float32 in [0.0, 1.0]
        - offset_map: (2, H, W) float32 where [0] is delta_y, [1] is delta_x
        - instances_info: List of dicts with id, cy, cx, area, sigma
    """
    h, w = instance_mask.shape
    fg_mask = (instance_mask > 0).astype(np.float32)
    center_map = np.zeros((h, w), dtype=np.float32)
    offset_map = np.zeros((2, h, w), dtype=np.float32)
    instances_info: List[Dict[str, float]] = []

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        return fg_mask, center_map, offset_map, instances_info

    # Coordinate grids
    y_grid, x_grid = np.ogrid[:h, :w]

    for inst_id in unique_ids:
        obj_mask = (instance_mask == inst_id)
        ys, xs = np.where(obj_mask)
        area = len(ys)
        if area < min_area:
            continue

        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

        # Determine Gaussian sigma
        inst_sigma = sigma
        if adaptive_sigma:
            equiv_radius = np.sqrt(area / np.pi)
            inst_sigma = float(np.clip(equiv_radius / 3.0, 1.5, 6.0))

        # Render Gaussian center peak within 3*sigma bounding box
        radius = int(np.ceil(3.0 * inst_sigma))
        y_min = max(0, int(np.floor(cy - radius)))
        y_max = min(h, int(np.ceil(cy + radius + 1)))
        x_min = max(0, int(np.floor(cx - radius)))
        x_max = min(w, int(np.ceil(cx + radius + 1)))

        sub_y, sub_x = np.ogrid[y_min:y_max, x_min:x_max]
        dist_sq = (sub_y - cy) ** 2 + (sub_x - cx) ** 2
        gaussian_patch = np.exp(-0.5 * dist_sq / (inst_sigma ** 2)).astype(np.float32)

        # Max-blend Gaussian peaks
        center_map[y_min:y_max, x_min:x_max] = np.maximum(
            center_map[y_min:y_max, x_min:x_max], gaussian_patch
        )

        # Exact centroid peak value must be 1.0 at closest integer pixel
        cy_int = int(np.clip(np.round(cy), 0, h - 1))
        cx_int = int(np.clip(np.round(cx), 0, w - 1))
        center_map[cy_int, cx_int] = 1.0

        # Compute dense 2D offset vectors for all pixels inside this instance
        # delta_y = cy - y, delta_x = cx - x
        offset_map[0, ys, xs] = (cy - ys).astype(np.float32)
        offset_map[1, ys, xs] = (cx - xs).astype(np.float32)

        instances_info.append({
            "id": float(inst_id),
            "cy": cy,
            "cx": cx,
            "area": float(area),
            "sigma": inst_sigma,
        })

    return fg_mask, center_map, offset_map, instances_info


def generate_center_offset_batch(
    instance_masks: Union[np.ndarray, torch.Tensor],
    sigma: float = 2.0,
    min_area: int = 1,
    adaptive_sigma: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch target generation for PyTorch DataLoader collation or GPU transform.

    Parameters
    ----------
    instance_masks : np.ndarray or torch.Tensor
        Batch of instance masks of shape (B, H, W).
    sigma : float, default 2.0
        Standard deviation for Gaussian center peaks.
    min_area : int, default 1
        Minimum pixel count per source.
    adaptive_sigma : bool, default False
        Adaptive Gaussian radius based on area.
    device : torch.device, optional
        Target device for output tensors.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - fg_masks: (B, 1, H, W) float32
        - center_maps: (B, 1, H, W) float32
        - offset_maps: (B, 2, H, W) float32
    """
    if isinstance(instance_masks, torch.Tensor):
        masks_np = instance_masks.detach().cpu().numpy()
    else:
        masks_np = instance_masks

    b, h, w = masks_np.shape
    fg_list, center_list, offset_list = [], [], []

    for i in range(b):
        fg, c_map, o_map, _ = generate_center_offset_targets_numpy(
            masks_np[i],
            sigma=sigma,
            min_area=min_area,
            adaptive_sigma=adaptive_sigma,
        )
        fg_list.append(torch.from_numpy(fg).unsqueeze(0))
        center_list.append(torch.from_numpy(c_map).unsqueeze(0))
        offset_list.append(torch.from_numpy(o_map))

    fg_tensor = torch.stack(fg_list, dim=0).float()          # (B, 1, H, W)
    center_tensor = torch.stack(center_list, dim=0).float()  # (B, 1, H, W)
    offset_tensor = torch.stack(offset_list, dim=0).float()  # (B, 2, H, W)

    if device is not None:
        fg_tensor = fg_tensor.to(device)
        center_tensor = center_tensor.to(device)
        offset_tensor = offset_tensor.to(device)

    return fg_tensor, center_tensor, offset_tensor
