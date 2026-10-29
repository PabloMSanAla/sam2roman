"""
Fast astronomical boundary and edge generation for 3-class semantic segmentation.
Maps integer-labeled source segmentation maps into (Background, Interior, Boundary).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def generate_3class_boundary_mask(
    segmap: np.ndarray,
    boundary_width: int = 1,
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
    boundary_width : int, default 1
        Pixel width of the source boundary ring.

    Returns
    -------
    np.ndarray
        2D integer array with values in {0, 1, 2} matching segmap dimensions.
    """
    target = np.zeros_like(segmap, dtype=np.int64)
    fg = (segmap > 0)
    if not np.any(fg):
        return target

    # 1. Outer perimeter of all sources via morphological erosion
    eroded_fg = binary_erosion(fg, iterations=boundary_width)
    fg_edge = fg & (~eroded_fg)

    # 2. Inter-object contact boundaries (where distinct touching objects meet)
    # Fast 4-neighbor differential shift
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
