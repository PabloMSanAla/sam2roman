"""Pytest fixtures for sam2roman."""

import pytest
import numpy as np
from pathlib import Path
from sam2roman.utils.synthetic import generate_synthetic_roman_fits
from sam2roman.io.fits import FITSImage


@pytest.fixture(scope="session")
def sample_fits_path(tmp_path_factory) -> Path:
    """Create a temporary synthetic Roman FITS file for testing."""
    fn = tmp_path_factory.mktemp("astro_data") / "test_roman_wfi.fits"
    generate_synthetic_roman_fits(
        output_path=fn,
        shape=(128, 128),
        n_stars=8,
        n_galaxies=4,
        seed=123,
    )
    return fn


@pytest.fixture
def sample_fits_image(sample_fits_path) -> FITSImage:
    """Return a FITSImage instance loaded from sample FITS file."""
    return FITSImage.from_file(sample_fits_path)
