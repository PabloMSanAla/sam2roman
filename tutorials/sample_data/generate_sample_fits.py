"""
Generate a sample Roman Space Telescope WFI FITS file for tutorials.
"""

from pathlib import Path
from sam2roman.utils.synthetic import generate_synthetic_roman_fits


def main():
    data_dir = Path(__file__).resolve().parent
    output_fits = data_dir / "sample_roman_wfi.fits"

    print(f"Generating realistic Roman WFI sample FITS image at: {output_fits}")
    image, header, catalog, segmap = generate_synthetic_roman_fits(
        output_path=output_fits,
        shape=(384, 384),
        n_stars=30,
        n_galaxies=18,
        sky_background=1.5,
        read_noise=2.0,
        seed=2026,
        ra_center=150.119,
        dec_center=2.2058,
        pixel_scale_arcsec=0.11,
        filter_name="F158",
    )
    print(f"Success! Created {output_fits.name} with shape {image.shape} and {len(catalog)} ground truth sources.")


if __name__ == "__main__":
    main()
