"""
Synthetic astronomical FITS image generator mimicking Roman Space Telescope WFI observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import numpy as np
from astropy.io import fits


def make_gaussian_psf(
    x0: float,
    y0: float,
    fwhm: float,
    amplitude: float,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Render a 2D Gaussian point source."""
    h, w = shape
    sigma = fwhm / 2.35482
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-0.5 * dist_sq / (sigma ** 2))


def make_sersic_galaxy(
    x0: float,
    y0: float,
    r_eff: float,
    amplitude: float,
    n_sersic: float,
    axis_ratio: float,
    pa_deg: float,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Render an elliptical Sérsic profile galaxy."""
    h, w = shape
    theta = np.deg2rad(pa_deg)
    y, x = np.ogrid[:h, :w]
    dx = x - x0
    dy = y - y0

    # Rotate coordinates
    x_rot = dx * np.cos(theta) + dy * np.sin(theta)
    y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

    # Elliptical radius
    b_n = 2.0 * n_sersic - 1.0 / 3.0 + 4.0 / (405.0 * n_sersic)
    r_ell = np.sqrt(x_rot ** 2 + (y_rot / max(axis_ratio, 0.1)) ** 2)
    profile = amplitude * np.exp(-b_n * ((r_ell / max(r_eff, 0.5)) ** (1.0 / n_sersic) - 1.0))
    return profile


def generate_synthetic_roman_fits(
    output_path: Optional[Union[str, Path]] = None,
    shape: Tuple[int, int] = (512, 512),
    n_stars: int = 25,
    n_galaxies: int = 15,
    sky_background: float = 1.5,
    read_noise: float = 2.0,
    seed: int = 42,
    ra_center: float = 150.119,
    dec_center: float = 2.2058,
    pixel_scale_arcsec: float = 0.11,  # Roman WFI pixel scale ~ 0.11 arcsec/pixel
    filter_name: str = "F158",
) -> Tuple[np.ndarray, fits.Header, List[Dict[str, float]], np.ndarray]:
    """
    Generate a realistic synthetic Roman Space Telescope WFI FITS image.

    Parameters
    ----------
    output_path : Optional[str or Path]
        If provided, writes the HDUList directly to this path on disk.
    shape : Tuple[int, int], default (512, 512)
        Pixel dimensions (Height, Width).
    n_stars : int, default 25
        Number of point sources (stars).
    n_galaxies : int, default 15
        Number of extended sources (galaxies).
    sky_background : float, default 1.5
        Mean sky background counts per second.
    read_noise : float, default 2.0
        Gaussian detector read noise (e-).
    seed : int, default 42
        Random seed for reproducibility.
    ra_center : float, default 150.119
        Right Ascension pointing center (degrees, e.g. COSMOS field).
    dec_center : float, default 2.2058
        Declination pointing center (degrees).
    pixel_scale_arcsec : float, default 0.11
        Plate scale in arcseconds per pixel (Roman WFI is 0.11 "/px).
    filter_name : str, default 'F158'
        Roman optical/NIR filter (e.g., 'F062', 'F106', 'F129', 'F158', 'F184', 'F213').

    Returns
    -------
    Tuple[np.ndarray, fits.Header, List[Dict], np.ndarray]
        (image_data, header, catalog, ground_truth_segmentation_map)
    """
    rng = np.random.RandomState(seed)
    h, w = shape

    # Initialize background
    image = np.full((h, w), sky_background, dtype=np.float32)
    # Poisson noise
    image = rng.poisson(np.clip(image, 0, None)).astype(np.float32)
    # Read noise
    image += rng.normal(0.0, read_noise, (h, w)).astype(np.float32)

    gt_segmap = np.zeros((h, w), dtype=np.int32)
    catalog: List[Dict[str, float]] = []
    object_id = 1

    # 1. Place Stars (PSFs)
    # Roman WFI PSF FWHM ~ 1.5 - 2.0 pixels at 1.5 microns
    for _ in range(n_stars):
        x0 = rng.uniform(w * 0.08, w * 0.92)
        y0 = rng.uniform(h * 0.08, h * 0.92)
        amplitude = 10.0 ** rng.uniform(2.2, 4.5)  # dynamic range: ~150 to ~30,000 counts
        fwhm = rng.uniform(1.4, 2.0)

        star_stamp = make_gaussian_psf(x0, y0, fwhm, amplitude, (h, w))
        image += star_stamp

        # Ground truth mask thresholded at 3-sigma noise
        star_mask = star_stamp > (3.0 * read_noise)
        if np.any(star_mask):
            gt_segmap[star_mask] = object_id
            catalog.append({
                "id": object_id,
                "type": "star",
                "x": float(x0),
                "y": float(y0),
                "flux": float(star_stamp.sum()),
                "fwhm": float(fwhm),
            })
            object_id += 1

    # 2. Place Galaxies (Sérsic profiles)
    for _ in range(n_galaxies):
        x0 = rng.uniform(w * 0.1, w * 0.9)
        y0 = rng.uniform(h * 0.1, h * 0.9)
        amplitude = 10.0 ** rng.uniform(1.8, 3.8)
        r_eff = rng.uniform(3.5, 16.0)  # half-light radius in pixels
        n_sersic = rng.choice([1.0, 1.5, 2.0, 4.0])  # disk vs spheroid
        axis_ratio = rng.uniform(0.3, 0.95)
        pa_deg = rng.uniform(0.0, 180.0)

        gal_stamp = make_sersic_galaxy(
            x0, y0, r_eff, amplitude, n_sersic, axis_ratio, pa_deg, (h, w)
        )
        image += gal_stamp

        gal_mask = gal_stamp > (2.5 * read_noise)
        if np.any(gal_mask):
            gt_segmap[gal_mask] = object_id
            catalog.append({
                "id": object_id,
                "type": "galaxy",
                "x": float(x0),
                "y": float(y0),
                "flux": float(gal_stamp.sum()),
                "r_eff": float(r_eff),
                "n_sersic": float(n_sersic),
            })
            object_id += 1

    # Build Header with standard Astronomical WCS
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = w
    header["NAXIS2"] = h
    header["TELESCOP"] = "ROMAN"
    header["INSTRUME"] = "WFI"
    header["FILTER"] = filter_name
    header["BUNIT"] = "ELECTRONS/S"

    # WCS
    header["WCSAXES"] = 2
    header["CRPIX1"] = w / 2.0
    header["CRPIX2"] = h / 2.0
    header["CRVAL1"] = ra_center
    header["CRVAL2"] = dec_center
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    cdelt = pixel_scale_arcsec / 3600.0  # degrees per pixel
    header["CD1_1"] = -cdelt
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = cdelt
    header["RADESYS"] = "ICRS"
    header["EQUINOX"] = 2000.0

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        primary_hdu = fits.PrimaryHDU(data=image.astype(np.float32), header=header)
        primary_hdu.writeto(out_p, overwrite=True)

    return image, header, catalog, gt_segmap
