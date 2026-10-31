"""
Astronomical ASDF file handling and preprocessing for Roman Space Telescope L2 data.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import numpy as np
from PIL import Image

from sam2roman.io.fits import stretch_astro_image

logger = logging.getLogger(__name__)

try:
    import asdf
    HAS_ASDF = True
except ImportError:
    HAS_ASDF = False


class ASDFImage:
    """
    Representation of an astronomical ASDF image (such as Roman Space Telescope L2 data).

    Handles Roman L2 ASDF schema, extracting 2D flux data, WCS coordinates,
    astronomical stretching (leakySB), and conversion to PIL / RGB formats.

    Parameters
    ----------
    data : np.ndarray
        2D astronomical flux array.
    meta : dict, optional
        Metadata dictionary extracted from ASDF tree.
    filepath : str or Path, optional
        Source filepath on disk.
    """

    def __init__(
        self,
        data: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
        filepath: Optional[Union[str, Path]] = None,
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.meta = meta if meta is not None else {}
        self.filepath = Path(filepath) if filepath is not None else None

        # Parse WCS if available in metadata (gwcs or astropy WCS)
        self.wcs = self.meta.get("wcs", None)

    @classmethod
    def from_file(
        cls,
        filepath: Union[str, Path],
    ) -> "ASDFImage":
        """
        Load an astronomical ASDF image from disk.

        Extracts the 2D science array from standard Roman L2 paths:
        `af['roman']['data']`, `af['data']`, or `af['science']`.

        Parameters
        ----------
        filepath : str or Path
            Path to the .asdf file.
        """
        if not HAS_ASDF:
            raise ImportError(
                "The 'asdf' package is required to load ASDF files. "
                "Install it via: pip install asdf"
            )

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"ASDF file not found at: {filepath}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with asdf.open(
                filepath,
                ignore_unrecognized_tag=True,
                ignore_missing_extensions=True,
            ) as af:
                tree = af.tree
                data = None
                meta: Dict[str, Any] = {}

                # Check Roman L2 structure
                if "roman" in tree and isinstance(tree["roman"], dict):
                    roman_dict = tree["roman"]
                    if "data" in roman_dict:
                        data = np.asarray(roman_dict["data"], dtype=np.float32)
                    meta = dict(roman_dict.get("meta", {}))
                elif "data" in tree:
                    data = np.asarray(tree["data"], dtype=np.float32)
                    meta = dict(tree.get("meta", {}))
                elif "science" in tree:
                    data = np.asarray(tree["science"], dtype=np.float32)
                    meta = dict(tree.get("meta", {}))
                else:
                    # Recursive search for 2D array
                    for key, val in tree.items():
                        if hasattr(val, "shape") and len(val.shape) == 2:
                            data = np.asarray(val, dtype=np.float32)
                            break

                if data is None:
                    available = list(tree.keys())
                    raise ValueError(
                        f"Could not locate 2D science data in ASDF file: {filepath.name}. "
                        f"Available top-level keys: {available}"
                    )

                data = np.squeeze(data)
                if data.ndim != 2:
                    raise ValueError(
                        f"Expected 2D array in ASDF file, got shape {data.shape} in {filepath.name}"
                    )

        return cls(data=data, meta=meta, filepath=filepath)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Dimensions (H, W) of the image data."""
        return self.data.shape

    def pixel_to_sky(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Convert pixel coordinates (x, y) to celestial coordinates (RA, Dec in degrees).

        Supports gwcs and astropy WCS models.
        """
        if self.wcs is None:
            return None
        try:
            if callable(self.wcs):
                # gwcs callable WCS(x, y)
                ra, dec = self.wcs(x, y)
                return float(ra), float(dec)
            elif hasattr(self.wcs, "pixel_to_world_values"):
                ra, dec = self.wcs.pixel_to_world_values(x, y)
                return float(ra), float(dec)
        except Exception as e:
            logger.debug(f"Failed to convert pixel ({x}, {y}) to sky: {e}")
        return None

    def sky_to_pixel(self, ra: float, dec: float) -> Optional[Tuple[float, float]]:
        """
        Convert celestial coordinates (RA, Dec in degrees) to pixel coordinates (x, y).
        """
        if self.wcs is None:
            return None
        try:
            if hasattr(self.wcs, "backward_run"):
                x, y = self.wcs.backward_run(ra, dec)
                return float(x), float(y)
            elif hasattr(self.wcs, "world_to_pixel_values"):
                x, y = self.wcs.world_to_pixel_values(ra, dec)
                return float(x), float(y)
        except Exception as e:
            logger.debug(f"Failed to convert sky ({ra}, {dec}) to pixel: {e}")
        return None

    def to_rgb(
        self,
        stretch_type: str = "leakysb",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        asinh_a: float = 0.05,
        zero_points: float = 31.4,
        pixel_scale: float = 0.11,
        mmin: float = 13.0,
        mmax: float = 32.0,
    ) -> np.ndarray:
        """
        Convert 2D astronomical image data to a 3-channel (H, W, 3) uint8 RGB array.
        """
        norm = stretch_astro_image(
            self.data,
            stretch_type=stretch_type,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            asinh_a=asinh_a,
            zero_points=zero_points,
            pixel_scale=pixel_scale,
            mmin=mmin,
            mmax=mmax,
        )
        uint8_data = (norm * 255.0).astype(np.uint8)
        return np.stack([uint8_data, uint8_data, uint8_data], axis=-1)

    def to_pil(
        self,
        stretch_type: str = "leakysb",
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
        asinh_a: float = 0.05,
        zero_points: float = 31.4,
        pixel_scale: float = 0.11,
        mmin: float = 13.0,
        mmax: float = 32.0,
    ) -> Image.Image:
        """
        Convert astronomical data to a PIL RGB Image ready for vision encoders.
        """
        rgb_arr = self.to_rgb(
            stretch_type=stretch_type,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            asinh_a=asinh_a,
            zero_points=zero_points,
            pixel_scale=pixel_scale,
            mmin=mmin,
            mmax=mmax,
        )
        return Image.fromarray(rgb_arr)
