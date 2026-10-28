"""Unit tests for FITS I/O and astronomical stretching."""

import numpy as np
import pytest
from astropy.io import fits
from sam2roman.io.fits import FITSImage, stretch_astro_image


def test_stretch_astro_image_shapes_and_bounds():
    data = np.random.normal(10.0, 2.0, (64, 64)).astype(np.float32)
    # Add a bright star
    data[30:35, 30:35] += 500.0

    for s_type in ["leakysb", "asinh", "linear", "log", "sqrt", "zscale"]:
        stretched = stretch_astro_image(data, stretch_type=s_type)
        assert stretched.shape == (64, 64)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0


def test_fits_image_loading_and_rgb(sample_fits_path):
    fits_img = FITSImage.from_file(sample_fits_path)
    assert fits_img.shape == (128, 128)
    assert fits_img.data.dtype == np.float32

    # Default stretch should now be leakysb
    rgb = fits_img.to_rgb()
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8
    # Verify same channel across R, G, and B
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])

    pil_img = fits_img.to_pil()
    assert pil_img.size == (128, 128)
    assert pil_img.mode == "RGB"


def test_wcs_coordinate_transforms(sample_fits_image):
    assert sample_fits_image.wcs is not None
    # Center pixel
    sky = sample_fits_image.pixel_to_sky(64.0, 64.0)
    assert sky is not None
    ra, dec = sky
    assert isinstance(ra, float)
    assert isinstance(dec, float)

    # Round trip
    px_back = sample_fits_image.sky_to_pixel(ra, dec)
    assert px_back is not None
    np.testing.assert_allclose(px_back, (64.0, 64.0), atol=1e-3)


def test_save_masks_to_fits(sample_fits_image, tmp_path):
    out_file = tmp_path / "masks_output.fits"
    m1 = np.zeros((128, 128), dtype=bool)
    m1[10:20, 10:20] = True
    m2 = np.zeros((128, 128), dtype=bool)
    m2[40:50, 40:50] = True

    sample_fits_image.save_masks_to_fits([m1, m2], output_path=out_file)
    assert out_file.exists()

    with fits.open(out_file) as hdul:
        assert len(hdul) >= 2
        assert "SEGMAP" in hdul
        segmap = hdul["SEGMAP"].data
        assert np.sum(segmap == 1) == 100
        assert np.sum(segmap == 2) == 100
