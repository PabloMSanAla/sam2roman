"""
Astronomical FITS file handling, preprocessing, and stretching for SAM 2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image


def stretch_astro_image(
    data: np.ndarray,
    stretch_type: str = "asinh",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    asinh_a: float = 0.05,
    log_a: float = 1000.0,
) -> np.ndarray:
    """
    Stretch a 2D astronomical array into normalized [0.0, 1.0] float32 values.

    Parameters
    ----------
    data : np.ndarray
        2D astronomical data array.
    stretch_type : str, default 'asinh'
        Type of stretch: 'asinh', 'zscale', 'percentile', 'linear', 'log', or 'sqrt'.
    lower_percentile : float, default 1.0
        Lower percentile for background clipping.
    upper_percentile : float, default 99.5
        Upper percentile for peak clipping.
    asinh_a : float, default 0.05
        Softening parameter for asinh stretch (lower values show fainter structures).
    log_a : float, default 1000.0
        Scale factor for log stretch.

    Returns
    -------
    np.ndarray
        Normalized 2D array with values in [0.0, 1.0].
    """
    img = np.array(data, dtype=np.float32, copy=True)
    # Replace non-finite values (NaN, Inf) with median or zero
    valid_mask = np.isfinite(img)
    if not np.any(valid_mask):
        return np.zeros_like(img, dtype=np.float32)

    bg_val = float(np.nanmedian(img))
    img[~valid_mask] = bg_val

    # Percentile clipping
    vmin = float(np.nanpercentile(img, lower_percentile))
    vmax = float(np.nanpercentile(img, upper_percentile))

    if vmax <= vmin:
        vmax = vmin + 1e-5

    # Clip range
    clipped = np.clip(img, vmin, vmax)

    # Normalize to [0, 1] before non-linear stretch
    norm = (clipped - vmin) / (vmax - vmin)

    stretch_type = stretch_type.lower()
    if stretch_type == "linear":
        stretched = norm
    elif stretch_type == "asinh":
        # asinh(x / a) / asinh(1 / a)
        stretched = np.arcsinh(norm / asinh_a) / np.arcsinh(1.0 / asinh_a)
    elif stretch_type == "log":
        stretched = np.log10(1.0 + log_a * norm) / np.log10(1.0 + log_a)
    elif stretch_type == "sqrt":
        stretched = np.sqrt(norm)
    elif stretch_type in ("zscale", "percentile"):
        stretched = norm
    else:
        raise ValueError(
            f"Unsupported stretch_type: {stretch_type}. Choose 'asinh', 'linear', 'log', 'sqrt', or 'zscale'."
        )

    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


class FITSImage:
    """
    Representation of an astronomical FITS image prepared for SAM 2 input.

    Handles multi-extension FITS, WCS coordinates, astronomical stretching,
    and conversion to PIL / RGB formats expected by SAM 2 vision encoders.
    """

    def __init__(
        self,
        data: np.ndarray,
        header: Optional[fits.Header] = None,
        filepath: Optional[Union[str, Path]] = None,
        extension: Union[int, str] = 0,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.header = header if header is not None else fits.Header()
        self.filepath = Path(filepath) if filepath is not None else None
        self.extension = extension

        # Parse WCS if header contains WCS keywords
        try:
            self.wcs = WCS(self.header) if self.header else None
        except Exception:
            self.wcs = None

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
        extension: Union[int, str] = 0,
    ) -> "FITSImage":
        """
        Load an astronomical FITS image from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the .fits or .fits.gz file.
        extension : int or str, default 0
            HDU index or extension name (e.g., 0, 1, 'SCI', 'PRIMARY').
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found at: {filepath}")

        with fits.open(filepath) as hdul:
            # If default extension is 0 but it's empty and HDU 1 exists (common in JWST/Roman), use HDU 1
            if extension == 0 and len(hdul) > 1 and hdul[0].data is None and hdul[1].data is not None:
                extension = 1

            hdu = hdul[extension]
            if hdu.data is None:
                raise ValueError(
                    f"HDU {extension} in {filepath.name} contains no image data. Available extensions: "
                    f"{[h.name for h in hdul]}"
                )
            data = np.squeeze(hdu.data)
            header = hdu.header.copy()

        return cls(data=data, header=header, filepath=filepath, extension=extension)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Dimensions (H, W) of the image data."""
        return self.data.shape

    def to_rgb(
        self,
        stretch_type: str = "asinh",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        asinh_a: float = 0.05,
    ) -> np.ndarray:
        """
        Convert 2D astronomical image data to a 3-channel (H, W, 3) uint8 RGB array.
        SAM 2 expects 3-channel RGB image format.

        Returns
        -------
        np.ndarray
            RGB array of shape (H, W, 3) with dtype uint8 [0, 255].
        """
        norm = stretch_astro_image(
            self.data,
            stretch_type=stretch_type,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            asinh_a=asinh_a,
        )
        uint8_data = (norm * 255.0).astype(np.uint8)
        # Duplicate single-band channel across 3 RGB channels
        rgb_data = np.stack([uint8_data, uint8_data, uint8_data], axis=-1)
        return rgb_data

    def to_pil(
        self,
        stretch_type: str = "asinh",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        asinh_a: float = 0.05,
    ) -> Image.Image:
        """
        Convert astronomical data to a PIL RGB Image ready for Hugging Face SAM 2 Processor.
        """
        rgb_arr = self.to_rgb(
            stretch_type=stretch_type,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            asinh_a=asinh_a,
        )
        return Image.fromarray(rgb_arr, mode="RGB")

    def pixel_to_sky(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinate (x, y) to Right Ascension (RA) and Declination (Dec) in degrees.
        """
        if self.wcs is not None and self.wcs.has_celestial:
            coord = self.wcs.pixel_to_world(x, y)
            return float(coord.ra.deg), float(coord.dec.deg)
        return None

    def sky_to_pixel(self, ra: float, dec: float) -> Optional[Tuple[float, float]]:
        """
        Convert RA and Dec (degrees) to pixel coordinates (x, y).
        """
        if self.wcs is not None and self.wcs.has_celestial:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            x, y = self.wcs.world_to_pixel(coord)
            return float(x), float(y)
        return None

    def save_masks_to_fits(
        self,
        masks: List[np.ndarray],
        output_path: Union[str, Path],
        labels: Optional[List[dict]] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Save segmentation masks back to a multi-extension FITS file,
        preserving the original primary image and celestial WCS headers.

        Parameters
        ----------
        masks : List[np.ndarray]
            List of 2D boolean or uint8 segmentation masks.
        output_path : str or Path
            Path to write the resulting FITS file.
        labels : Optional[List[dict]]
            Optional metadata/attributes for each mask (e.g. area, iou_score).
        overwrite : bool, default True
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        primary_hdu = fits.PrimaryHDU(data=self.data, header=self.header)
        primary_hdu.header["COMMENT"] = "Primary astronomical image processed with sam2roman"

        hdul = [primary_hdu]

        # Combine into labeled segmentation map where 0 is background and i is mask i
        segmap = np.zeros_like(self.data, dtype=np.int32)
        for i, m in enumerate(masks, start=1):
            segmap[m.astype(bool)] = i

        seg_hdu = fits.ImageHDU(data=segmap, name="SEGMAP")
        seg_hdu.header["COMMENT"] = "Combined integer segmentation label map (0=bg, >0=object_id)"
        hdul.append(seg_hdu)

        fits_hdulist = fits.HDUList(hdul)
        fits_hdulist.writeto(output_path, overwrite=overwrite)
