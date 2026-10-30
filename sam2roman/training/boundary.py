"""
Fast astronomical boundary, edge, and distance transform generation for 3-class segmentation.
Maps integer-labeled source segmentation maps into (Background, Interior, Boundary).
Supports both discrete masks, continuous soft-boundary probability fields, and signed distance fields (SDF).
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, distance_transform_edt


def generate_3class_boundary_mask(
    segmap: np.ndarray,
    boundary_width: int = 2,
    soften: bool = False,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Convert an integer-labeled astronomical segmentation map into a 3-class target:
    - Class 0: Sky Background
    - Class 1: Source Interior / Core
    - Class 2: Source Boundary / Perimeter & Inter-Object Touching Interfaces

    Parameters
    ----------
    segmap : np.ndarray
        2D integer array where 0 is sky background and 1..N are discrete source IDs.
    boundary_width : int, default 2
        Pixel width of the source boundary ring (default 2px ensures edges survive downsampling).
    soften : bool, default False
        If True, returns a continuous 3-channel probability map (3, H, W) via Gaussian smoothing.
        If False, returns a discrete 2D integer array (H, W) with values in {0, 1, 2}.
    sigma : float, default 1.2
        Gaussian filter standard deviation used when soften=True.

    Returns
    -------
    np.ndarray
        If soften=False: 2D int64 array of shape (H, W) with values in {0, 1, 2}.
        If soften=True:  3D float32 array of shape (3, H, W) where sum across axis 0 equals 1.0.
    """
    if soften:
        return generate_soft_boundary_target(segmap, boundary_width=boundary_width, sigma=sigma)

    target = np.zeros_like(segmap, dtype=np.int64)
    fg = (segmap > 0)
    if not np.any(fg):
        return target

    # 1. Outer perimeter of all sources via morphological erosion
    eroded_fg = binary_erosion(fg, iterations=boundary_width)
    fg_edge = fg & (~eroded_fg)

    # 2. Inter-object contact boundaries (where distinct touching objects meet)
    shift_up = np.pad(segmap[1:, :], ((0, 1), (0, 0)), mode="edge")
    shift_down = np.pad(segmap[:-1, :], ((1, 0), (0, 0)), mode="edge")
    shift_left = np.pad(segmap[:, 1:], ((0, 0), (0, 1)), mode="edge")
    shift_right = np.pad(segmap[:, :-1], ((0, 0), (1, 0)), mode="edge")

    touching_interface = (
        ((shift_up != segmap) & (shift_up > 0) & (segmap > 0))
        | ((shift_down != segmap) & (shift_down > 0) & (segmap > 0))
        | ((shift_left != segmap) & (shift_left > 0) & (segmap > 0))
        | ((shift_right != segmap) & (shift_right > 0) & (segmap > 0))
    )

    boundary = fg_edge | touching_interface
    interior = fg & (~boundary)

    target[interior] = 1
    target[boundary] = 2

    # 3. Ensure tiny astronomical sources (e.g. 1-4 pixels) do not lose their interior core
    if np.any(boundary):
        unique_lbls = np.unique(segmap[boundary])
        for lbl in unique_lbls:
            if lbl == 0:
                continue
            obj_pixels = (segmap == lbl)
            if not np.any(obj_pixels & (target == 1)):
                ys, xs = np.where(obj_pixels)
                cx = int(round(float(np.mean(xs))))
                cy = int(round(float(np.mean(ys))))
                cx = min(max(cx, 0), segmap.shape[1] - 1)
                cy = min(max(cy, 0), segmap.shape[0] - 1)
                if segmap[cy, cx] == lbl:
                    target[cy, cx] = 1
                else:
                    target[ys[0], xs[0]] = 1

    return target


def generate_soft_boundary_target(
    segmap: np.ndarray,
    boundary_width: int = 2,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Generate a 3-channel continuous probability map (3, H, W) where the boundary
    channel is smoothed using a Gaussian kernel (Option A: Soft Boundary Targets).

    Parameters
    ----------
    segmap : np.ndarray
        2D integer array where 0 is sky background and 1..N are discrete source IDs.
    boundary_width : int, default 2
        Pixel width of initial boundary contour before smoothing.
    sigma : float, default 1.2
        Gaussian standard deviation for boundary smoothing.

    Returns
    -------
    np.ndarray
        Continuous float32 array of shape (3, H, W) normalized so that
        channel 0 (sky) + channel 1 (interior) + channel 2 (boundary) == 1.0.
    """
    h, w = segmap.shape
    target = np.zeros((3, h, w), dtype=np.float32)
    fg = (segmap > 0)
    if not np.any(fg):
        target[0] = 1.0
        return target

    # Extract binary boundary
    eroded_fg = binary_erosion(fg, iterations=boundary_width)
    fg_edge = fg & (~eroded_fg)

    shift_up = np.pad(segmap[1:, :], ((0, 1), (0, 0)), mode="edge")
    shift_down = np.pad(segmap[:-1, :], ((1, 0), (0, 0)), mode="edge")
    shift_left = np.pad(segmap[:, 1:], ((0, 0), (0, 1)), mode="edge")
    shift_right = np.pad(segmap[:, :-1], ((0, 0), (1, 0)), mode="edge")
    touching_interface = (
        ((shift_up != segmap) & (shift_up > 0) & (segmap > 0))
        | ((shift_down != segmap) & (shift_down > 0) & (segmap > 0))
        | ((shift_left != segmap) & (shift_left > 0) & (segmap > 0))
        | ((shift_right != segmap) & (shift_right > 0) & (segmap > 0))
    )
    edge_binary = (fg_edge | touching_interface).astype(np.float32)

    # Gaussian smoothing to create continuous edge probability field
    soft_edge = gaussian_filter(edge_binary, sigma=sigma)
    if soft_edge.max() > 0:
        soft_edge = np.clip(soft_edge / soft_edge.max(), 0.0, 1.0)

    is_interior = (fg & (~(edge_binary > 0.5))).astype(np.float32)
    is_bg = (~fg).astype(np.float32)

    # Assign smoothly: boundary probability takes precedence, remaining mass split between core/bg
    soft_int = (1.0 - soft_edge) * is_interior
    soft_bg = (1.0 - soft_edge) * is_bg

    target[0] = soft_bg
    target[1] = soft_int
    target[2] = soft_edge

    # Ensure exact probability simplex sum = 1.0 at every pixel
    total = np.maximum(target.sum(axis=0), 1e-6)
    target = (target / total).astype(np.float32)
    return target


def generate_distance_transform_target(
    segmap: np.ndarray,
    max_distance: float = 10.0,
    truncate: bool = True,
) -> np.ndarray:
    """
    Generate continuous Signed Distance Field (SDF) from source boundaries (Option B).
    Positive values represent distance inside source; negative values represent distance in background.

    Parameters
    ----------
    segmap : np.ndarray
        2D integer array with source IDs.
    max_distance : float, default 10.0
        Maximum distance truncation scale.
    truncate : bool, default True
        If True, clips distances to [-1.0, 1.0].

    Returns
    -------
    np.ndarray
        2D float32 array of shape (H, W).
    """
    fg = (segmap > 0)
    if not np.any(fg):
        return np.full_like(segmap, -1.0 if truncate else -max_distance, dtype=np.float32)

    dist_in = distance_transform_edt(fg)
    dist_out = distance_transform_edt(~fg)
    sdf = (dist_in - dist_out).astype(np.float32)

    if truncate and max_distance > 0:
        sdf = np.clip(sdf / max_distance, -1.0, 1.0)
    return sdf
