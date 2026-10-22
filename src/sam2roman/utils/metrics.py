"""
Astronomical source metrics and photometry on SAM 2 masks.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np


def compute_source_photometry(
    data: np.ndarray,
    masks: List[np.ndarray],
    bkg_median: Optional[float] = None,
    bkg_std: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Compute astronomical photometric properties inside each SAM 2 mask.

    Parameters
    ----------
    data : np.ndarray
        Raw 2D astronomical image data (counts / flux).
    masks : List[np.ndarray]
        List of 2D boolean masks.
    bkg_median : float, optional
        Sky background median count. Estimated from data if None.
    bkg_std : float, optional
        Sky background noise sigma. Estimated from data if None.

    Returns
    -------
    List[Dict[str, Any]]
        Catalog of detected sources with area, raw_flux, net_flux, peak_flux, snr, etc.
    """
    if bkg_median is None or bkg_std is None:
        # 3-sigma clipped background estimation
        med = float(np.median(data))
        std = float(np.std(data))
        clip_mask = np.abs(data - med) < 3.0 * std
        if np.any(clip_mask):
            bkg_median = float(np.median(data[clip_mask]))
            bkg_std = float(np.std(data[clip_mask]))
        else:
            bkg_median = med
            bkg_std = std

    catalog = []
    for idx, mask in enumerate(masks, start=1):
        m = mask.astype(bool)
        area = int(np.sum(m))
        if area == 0:
            continue

        pixels = data[m]
        raw_flux = float(np.sum(pixels))
        net_flux = raw_flux - (bkg_median * area)
        peak_val = float(np.max(pixels))
        snr = net_flux / (bkg_std * np.sqrt(area) + 1e-8)

        ys, xs = np.where(m)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        # Approximate half-light radius
        r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        sort_r = np.sort(r)
        cum_flux = np.cumsum(np.sort(pixels)[::-1])
        half_idx = np.searchsorted(cum_flux, 0.5 * raw_flux)
        r_half = float(sort_r[min(half_idx, len(sort_r) - 1)])

        catalog.append({
            "source_id": idx,
            "x_centroid": cx,
            "y_centroid": cy,
            "area_pixels": area,
            "raw_flux": raw_flux,
            "net_flux": net_flux,
            "peak_flux": peak_val,
            "snr": snr,
            "r_half_light": r_half,
        })

    return catalog
