"""Unit tests for synthetic Roman FITS scene generator."""

import numpy as np
from sam2roman.utils.synthetic import generate_synthetic_roman_fits


def test_generate_synthetic_roman_fits(tmp_path):
    out_file = tmp_path / "synthetic_roman.fits"
    image, header, catalog, segmap = generate_synthetic_roman_fits(
        output_path=out_file,
        shape=(64, 64),
        n_stars=5,
        n_galaxies=3,
        seed=42,
    )

    assert out_file.exists()
    assert image.shape == (64, 64)
    assert header["TELESCOP"] == "ROMAN"
    assert header["INSTRUME"] == "WFI"
    assert "CRVAL1" in header
    assert len(catalog) > 0
    assert segmap.shape == (64, 64)
    assert np.max(segmap) >= 1
