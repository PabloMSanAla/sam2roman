#!/usr/bin/env python
"""
Tutorial 01: Segment Everything in an Astronomical FITS Image with SAM 2 Out-of-the-Box
========================================================================================

This tutorial demonstrates how to:
1. Load a Roman Space Telescope / Astronomical FITS image with WCS metadata.
2. Apply astronomical non-linear stretching (asinh / zscale) to map dynamic range to RGB.
3. Initialize Meta's Segment Anything Model 2 (SAM 2) from Hugging Face Hub using PyTorch.
4. Run high-performance automatic mask generation ("segment everything") with verbose logging.
5. In a separate step, measure astronomical source photometry (flux, centroids, SNR).
6. Overlay segmentation masks and save publication-quality diagnostic plots.
7. Save the segmented object catalog and masks into a standard FITS multi-extension file.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import sam2roman core components
from sam2roman.io.fits import FITSImage
from sam2roman.inference.segmentor import SAM2Segmentor
from sam2roman.visualization.display import plot_detection_summary
from sam2roman.utils.synthetic import generate_synthetic_roman_fits


def main():
    print("=" * 70)
    print("   SAM 2 for Roman Space Telescope: Tutorial 01 - Segment Everything   ")
    print("=" * 70)

    # 1. Locate or create sample FITS file
    base_dir = Path(__file__).resolve().parent
    fits_path = base_dir / "sample_data" / "sample_roman_wfi.fits"
    fits_path = Path('/Users/pmsanch1/storage/ROMAN/comissioning/PID1083/detectors/fits/r0108301001001001001_WFI3.fits')

    if not fits_path.exists():
        print(f"\n[Step 1] Sample FITS not found at {fits_path.name}. Generating synthetic field...")
        generate_synthetic_roman_fits(
            output_path=fits_path,
            shape=(384, 384),
            n_stars=30,
            n_galaxies=18,
            seed=2026,
            filter_name="F158",
        )

    # 2. Load FITS image
    print(f"\n[Step 2] Loading FITS image from: {fits_path}")
    astro_img = FITSImage.from_file(fits_path)
    print(f"  -> Dimensions: {astro_img.shape[0]} x {astro_img.shape[1]} pixels")
    print(f"  -> Telescope:  {astro_img.header.get('TELESCOP', 'N/A')}")
    print(f"  -> Instrument: {astro_img.header.get('INSTRUME', 'N/A')}")
    print(f"  -> Filter:     {astro_img.header.get('FILTER', 'N/A')}")
    print(f"  -> Data Range: min={astro_img.data.min():.2f}, max={astro_img.data.max():.2f}, median={np.median(astro_img.data):.2f}")

    if astro_img.wcs:
        ra_c, dec_c = astro_img.pixel_to_sky(astro_img.shape[1] / 2, astro_img.shape[0] / 2)
        print(f"  -> WCS Center: RA = {ra_c:.5f} deg, Dec = {dec_c:.5f} deg")

    # 3. Initialize SAM 2 from Hugging Face
    model_id = "facebook/sam2.1-hiera-tiny"
    print(f"\n[Step 3] Initializing SAM 2 from Hugging Face: '{model_id}'")
    segmentor = SAM2Segmentor(model_id=model_id)
    print(f"  -> Execution Device: {segmentor.model_wrapper.device}")

    # 4. Run automatic segmentation ("Segment Everything") - Step 1: Segmentation ONLY
    print("\n[Step 4] Running 'Segment Everything' across the astronomical field (verbose mode)...")
    results = segmentor.segment_everything(
        image=astro_img,
        prompt_mode="astronomical",  # or 'grid'
        pred_iou_thresh=0.65,
        min_mask_region_area=12,    # Filter out hot pixels and cosmic rays
        stretch_type="leakysb",
        verbose=True,
    )
    print(f"  -> Successfully detected {len(results)} astronomical sources!")

    # 5. Measure astronomical source photometry - Step 2: Separate Photometry
    print("\n[Step 5] Separately measuring source photometry on detected masks...")
    catalog = results.compute_photometry(astro_img, verbose=True)

    print(f"\n{'ID':<5} {'Centroid (X, Y)':<18} {'Area (px)':<10} {'Net Flux (e-/s)':<16} {'Peak':<10} {'SNR':<8}")
    print("-" * 72)
    for src in catalog[:10]:
        print(
            f"{src['source_id']:<5} "
            f"({src['x_centroid']:5.1f}, {src['y_centroid']:5.1f})   "
            f"{src['area_pixels']:<10} "
            f"{src['net_flux']:<16.1f} "
            f"{src['peak_flux']:<10.1f} "
            f"{src['snr']:<8.1f}"
        )
    if len(catalog) > 10:
        print(f"... and {len(catalog) - 10} more sources.")

    # 6. Visualize results
    summary_plot_path = base_dir / "01_segmentation_summary.png"
    print(f"\n[Step 6] Generating 3-panel astronomical diagnostic summary...")
    fig = plot_detection_summary(
        image=astro_img,
        seg_result=results,
        title_prefix="Roman WFI (F158)",
        save_path=summary_plot_path,
    )
    plt.close(fig)
    print(f"  -> Saved summary visualization to: {summary_plot_path}")

    # 7. Save segmentation mask to FITS file
    output_seg_fits = base_dir / "sample_data" / "sample_roman_wfi_segmented.fits"
    print(f"\n[Step 7] Exporting labeled segmentation map to FITS: {output_seg_fits.name}")
    astro_img.save_masks_to_fits(results.masks, output_path=output_seg_fits)
    print(f"  -> Wrote multi-extension FITS file with primary SCI data and SEGMAP extension.")

    print("\n" + "=" * 70)
    print("   Tutorial 01 Completed Successfully!   ")
    print("=" * 70)


if __name__ == "__main__":
    main()
