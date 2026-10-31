"""
Unified reader for astronomical images (FITS and Roman ASDF).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

from sam2roman.io.fits import FITSImage
from sam2roman.io.asdf_io import ASDFImage


def load_astro_image(
    filepath: Union[str, Path],
    extension: Union[int, str] = 0,
) -> Union[FITSImage, ASDFImage]:
    """
    Load an astronomical image from disk, automatically detecting FITS or ASDF format.

    Parameters
    ----------
    filepath : str or Path
        Path to the astronomical image file (.fits, .fits.gz, .asdf, etc.).
    extension : int or str, default 0
        HDU extension index or name for FITS files.

    Returns
    -------
    FITSImage or ASDFImage
        Wrapped astronomical image object with .data, .wcs, .to_rgb(), and .to_pil() interfaces.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Astronomical image file not found at: {path}")

    # Check suffix
    name_lower = path.name.lower()
    if name_lower.endswith(".asdf"):
        return ASDFImage.from_file(path)
    elif name_lower.endswith((".fits", ".fits.gz", ".fit", ".fts")):
        return FITSImage.from_file(path, extension=extension)
    else:
        # Fallback: check file magic/header
        try:
            with open(path, "rb") as f:
                header_bytes = f.read(16)
            if header_bytes.startswith(b"#ASDF"):
                return ASDFImage.from_file(path)
        except Exception:
            pass

        # Try FITS by default
        return FITSImage.from_file(path, extension=extension)
