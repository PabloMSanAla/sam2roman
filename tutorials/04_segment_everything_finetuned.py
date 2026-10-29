#!/usr/bin/env python
"""
Tutorial 04: Segment Everything in a FITS Image Using Fine-Tuned Checkpoints
============================================================================

This script demonstrates how to:
1. Load fine-tuned model weights directly from `checkpoint_best.pt`.
2. Run promptless automatic mask generation ("segment everything") on a FITS image.
3. Compare prompt modes: 'astronomical' (fast peak detection) vs 'grid' (dense grid).
4. Measure astronomical source photometry (net flux, SNR, centroid, half-light radius).
5. Overlay segmentation masks and save diagnostic plots.
6. Export the segmented masks to a multi-extension FITS file with a SEGMAP extension.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sam2roman.io.fits import FITSImage
from sam2roman.inference.segmentor import SAM2Segmentor
from sam2roman.visualization.display import plot_detection_summary, plot_masks_overlay
from sam2roman.utils.synthetic import generate_synthetic_roman_fits


def main():
    print("=" * 75)
    print("   SAM 2 for Roman: Segment Everything with Fine-Tuned Checkpoint   ")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "sample_data"
    output_dir = base_dir / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate Fine-Tuned Checkpoint
    checkpoint_path = output_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        candidates = list(output_dir.glob("checkpoint_epoch_*.pt"))
        if candidates:
            checkpoint_path = sorted(candidates)[-1]
        else:
            raise FileNotFoundError(
                f"No fine-tuned checkpoint found in '{output_dir}'. "
                "Please run tutorial 02 or 03 first to produce a checkpoint."
            )

    print(f"\n[Step 1] Loading fine-tuned weights from: {checkpoint_path.name}")
    # Initialize segmentor directly from the checkpoint
    segmentor = SAM2Segmentor.from_checkpoint(checkpoint_path)
    print(f"  -> Execution Device: {segmentor.model_wrapper.device}")
    print(f"  -> Foundation Model:  {segmentor.model_wrapper.model_id}")

    # 2. Load Astronomical FITS Image
    fits_path = data_dir / "sample_roman_train_field.fits"
    if not fits_path.exists():
        fits_path = data_dir / "sample_roman_wfi.fits"

    if not fits_path.exists():
        print(f"\n[Step 2] Sample FITS not found. Generating synthetic Roman field...")
        generate_synthetic_roman_fits(
            output_path=fits_path,
            shape=(512, 512),
            n_stars=40,
            n_galaxies=25,
            seed=42,
            filter_name="F158",
        )

    print(f"\n[Step 2] Loading target FITS image: {fits_path.name}")
    astro_img = FITSImage.from_file(fits_path)
    print(f"  -> Dimensions: {astro_img.shape[0]} x {astro_img.shape[1]} pixels")
    print(f"  -> Flux stats: min={astro_img.data.min():.2f}, max={astro_img.data.max():.2f}, median={np.median(astro_img.data):.2f}")

    # 3. Run Segment Everything with the Fine-Tuned Model
    # 'astronomical' mode places prompt queries on local peak candidates (fast and memory-efficient)
    # 'grid' mode evaluates a uniform N x N grid of candidate points
    prompt_mode = "astronomical"
    print(f"\n[Step 3] Running 'segment_everything' (prompt_mode='{prompt_mode}')...")
    results = segmentor.segment_everything(
        image=astro_img,
        prompt_mode=prompt_mode,
        pred_iou_thresh=0.60,       # Predicted mask quality threshold
        min_mask_region_area=12,    # Rejects hot pixels / cosmic ray noise
        iou_overlap_thresh=0.70,    # Non-maximum suppression deduplication threshold
        stretch_type="leakysb",     # Default surface brightness stretch
        verbose=True,
    )
    print(f"  -> Fine-tuned model detected {len(results)} discrete astronomical sources!")

    # 4. Measure Source Photometry
    print(f"\n[Step 4] Measuring source photometry on detected masks...")
    catalog = results.compute_photometry(astro_img, verbose=True)
    if catalog:
        brightest = max(catalog, key=lambda s: s["net_flux"])
        print(f"  -> Brightest source: ID {brightest['source_id']}, Flux: {brightest['net_flux']:.1f}, SNR: {brightest['snr']:.1f}")

    # 5. Visualize and Save Diagnostic Plots
    summary_plot_path = base_dir / "04_finetuned_segmentation_summary.png"
    print(f"\n[Step 5] Saving 3-panel detection summary to: {summary_plot_path.name}")
    fig = plot_detection_summary(
        image=astro_img,
        seg_result=results,
        title_prefix=f"Fine-Tuned SAM 2 ({astro_img.shape[0]}x{astro_img.shape[1]})",
        save_path=summary_plot_path,
    )
    plt.close(fig)

    overlay_plot_path = base_dir / "04_finetuned_masks_overlay.png"
    fig_overlay = plot_masks_overlay(
        image=astro_img,
        masks=results.masks,
        boxes=results.boxes,
        centroids=results.centroids,
        stretch_type="leakysb",
        save_path=overlay_plot_path,
        title="Fine-Tuned SAM 2 Source Detections",
    )
    plt.close(fig_overlay)
    print(f"  -> Saved overlay plot to: {overlay_plot_path.name}")

    # 6. Export Masks to FITS Multi-Extension File
    segmented_fits_path = base_dir / "checkpoints" / f"{fits_path.stem}_finetuned_segmap.fits"
    print(f"\n[Step 6] Exporting masks to multi-extension FITS: {segmented_fits_path.name}")
    astro_img.save_masks_to_fits(
        masks=results.masks,
        output_path=segmented_fits_path,
        overwrite=True,
    )
    print(f"  -> Wrote SEGMAP HDU containing {len(results)} labeled source masks.")

    print("\n" + "=" * 75)
    print("   Tutorial 04 Completed Successfully!   ")
    print("=" * 75)


if __name__ == "__main__":
    main()
