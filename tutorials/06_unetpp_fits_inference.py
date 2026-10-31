#!/usr/bin/env python3
"""
Tutorial 06: Full-Image Astronomical Inference with Boundary-Aware UNet++
=========================================================================

This executable tutorial script demonstrates how to:
1. Load astronomical FITS (or Roman ASDF) imaging with celestial WCS metadata.
2. Apply the photometric Leaky Surface Brightness (leakySB) stretch.
3. Load a fine-tuned UNet++ model checkpoint with auto-detected hardware acceleration
   (Apple Silicon MPS, NVIDIA CUDA with TF32/AMP, or CPU).
4. Run full-image dense 3-class segmentation inference (predict_dense) using
   cosine-windowed sliding tiles.
5. Generate and save a publication-quality side-by-side 2-panel figure:
   - Panel 1: LeakySB Astronomical Image
   - Panel 2: Boundary-Aware 3-Class Predicted Segmentation Mask (Sky, Core, Edge)
6. Perform high-contrast cutout inspection of touching objects.
7. Extract discrete source instances via marker-controlled watershed deblending
   (segment_image) and export a multi-extension FITS file.

Usage:
    python tutorials/06_unetpp_fits_inference.py
    python tutorials/06_unetpp_fits_inference.py --fits path/to/image.fits --checkpoint path/to/unetpp_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Safeguard macOS dual-libomp conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import numpy as np
import torch
from astropy.io import fits

from sam2roman.io.reader import load_astro_image
from sam2roman.io.fits import FITSImage, stretch_astro_image
from sam2roman.inference.unetpp_segmentor import UNetPPSegmentor
from sam2roman.visualization.display import plot_two_panel_prediction, plot_masks_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UNet++ inference on astronomical FITS image and plot 2-panel results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fits",
        type=str,
        default=None,
        help="Path to astronomical image (.fits, .fits.gz, or .asdf). Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to UNet++ checkpoint (.pt). Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tile size in pixels for sliding window inference.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in pixels between adjacent tiles.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for concurrent tile inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tutorials",
        help="Directory to save generated plots and FITS products.",
    )
    return parser.parse_args()


def find_default_paths(base_dir: Path) -> tuple[Path, Path]:
    """Auto-discover checkpoint and FITS image."""
    ckpt_candidates = [
        base_dir.parent / "experiments/unet/test_1/unetpp_best.pt",
        Path("experiments/unet/test_1/unetpp_best.pt"),
        base_dir / "checkpoints/unetpp_best.pt",
        base_dir.parent / "checkpoints/unetpp_best.pt",
    ]
    ckpt_path = None
    for cand in ckpt_candidates:
        if cand.exists():
            ckpt_path = cand.resolve()
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "Could not find a UNet++ checkpoint. Specify with --checkpoint."
        )

    image_candidates = [
        base_dir / "sample_data/sample_roman_train_field.fits",
        Path("/Volumes/T7/storage/ROMAN/synthetic_romanisim_sam2/images/scene_0001_F106_detector_01.fits"),
        base_dir / "sample_data/sample_roman_wfi.fits",
    ]
    fits_path = None
    for cand in image_candidates:
        if cand.exists():
            fits_path = cand.resolve()
            break

    if fits_path is None:
        raise FileNotFoundError(
            "Could not find an astronomical FITS image. Specify with --fits."
        )

    return ckpt_path, fits_path


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    print("=" * 75)
    print("   Tutorial 06: Full-Image Astronomical Inference with UNet++   ")
    print("=" * 75)

    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"\n[Hardware] Accelerator: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  -> CUDA active: TF32=True, AMP (bfloat16) enabled")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n[Hardware] Accelerator: Apple Silicon MPS (Metal Performance Shaders active)")
    else:
        device = torch.device("cpu")
        print(f"\n[Hardware] Accelerator: CPU")

    # 2. Path Discovery
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        ckpt_path, _ = find_default_paths(base_dir)

    if args.fits:
        fits_path = Path(args.fits).resolve()
    else:
        _, fits_path = find_default_paths(base_dir)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Config] Checkpoint: {ckpt_path}")
    print(f"[Config] Image:      {fits_path}")
    print(f"[Config] Output Dir: {out_dir}")

    # 3. Ingest Image & Display Metadata
    astro_img = load_astro_image(fits_path)
    raw_data = astro_img.data
    h, w = raw_data.shape

    print("\n[Step 1] Ingested Astronomical Image:")
    print(f"  Dimensions:   {w} × {h} pixels ({w * h / 1e6:.2f} Megapixels)")
    print(f"  Dtype:        {raw_data.dtype}")
    print(f"  Flux Range:   [{np.nanmin(raw_data):.3f}, {np.nanmax(raw_data):.3f}]")
    print(f"  Flux Median:  {np.nanmedian(raw_data):.3f}")

    # 4. Initialize Segmentor
    print(f"\n[Step 2] Initializing UNetPPSegmentor from checkpoint...")
    segmentor = UNetPPSegmentor.from_checkpoint(ckpt_path, device=device)
    n_params = sum(p.numel() for p in segmentor.model.parameters())
    print(f"  Loaded model weights ({n_params:,} parameters) onto {segmentor.device}.")

    # 5. Full-Image Dense Inference
    print(f"\n[Step 3] Running predict_dense (tile_size={args.tile_size}, overlap={args.overlap})...")
    t0 = time.time()
    pred_mask, probs, stretched_img = segmentor.predict_dense(
        image=fits_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        stretch_type="leakysb",
        verbose=True,
    )
    dt = time.time() - t0
    total_px = pred_mask.size
    n_sky = int(np.sum(pred_mask == 0))
    n_core = int(np.sum(pred_mask == 1))
    n_edge = int(np.sum(pred_mask == 2))

    print(f"  Inference finished in {dt:.2f}s!")
    print(f"  Class 0 (Sky):   {n_sky:,} px ({100.0 * n_sky / total_px:.2f}%)")
    print(f"  Class 1 (Core):  {n_core:,} px ({100.0 * n_core / total_px:.2f}%)")
    print(f"  Class 2 (Edge):  {n_edge:,} px ({100.0 * n_edge / total_px:.2f}%)")

    # 6. Save 2-Panel Figure
    fig_path = out_dir / "06_unetpp_two_panel_prediction.png"
    print(f"\n[Step 4] Generating side-by-side 2-panel prediction figure...")
    fig = plot_two_panel_prediction(
        stretched=stretched_img,
        pred_mask=pred_mask,
        title=f"UNet++ Inference: {fits_path.name} (Checkpoint: {ckpt_path.name})",
        save_path=fig_path,
        figsize=(16, 8),
        dpi=150,
    )
    plt.close(fig)
    print(f"  Saved figure: {fig_path}")

    # 7. Extract Discrete Source Instances
    print(f"\n[Step 5] Extracting source instances via marker-controlled watershed...")
    res = segmentor.segment_image(
        image=fits_path,
        stretch_type="leakysb",
        interior_thresh=0.45,
        fg_thresh=0.50,
        min_area=10,
        verbose=True,
    )
    print(f"  Extracted {len(res.masks)} distinct astronomical sources.")

    # Save Overlay Figure
    if len(res.masks) > 0:
        overlay_path = out_dir / "06_unetpp_instances_overlay.png"
        fig_overlay = plot_masks_overlay(
            image=raw_data,
            masks=res.masks,
            boxes=res.boxes,
            centroids=res.centroids,
            stretch_type="leakysb",
            title=f"UNet++ Deblended Instances ({len(res.masks)} Sources) - {fits_path.name}",
            save_path=overlay_path,
            figsize=(10, 10),
        )
        plt.close(fig_overlay)
        print(f"  Saved instance overlay: {overlay_path}")

    # 8. Export Multi-Extension FITS
    segmap_fits_path = out_dir / "06_unetpp_predicted_segmap.fits"
    primary_header = astro_img.header if hasattr(astro_img, "header") and astro_img.header is not None else fits.Header()
    prim_hdu = fits.PrimaryHDU(data=raw_data, header=primary_header)

    seg_header = fits.Header()
    seg_header["EXTNAME"] = "PRED_MASK"
    seg_header["COMMENT"] = "UNet++ 3-Class Segmap: 0=Sky, 1=Core, 2=Boundary"
    seg_header["CHECKPT"] = ckpt_path.name
    seg_header["SRCCOUNT"] = len(res.masks)
    seg_hdu = fits.ImageHDU(data=pred_mask.astype(np.uint8), header=seg_header)

    hdul = fits.HDUList([prim_hdu, seg_hdu])
    hdul.writeto(segmap_fits_path, overwrite=True)
    print(f"\n[Step 6] Exported multi-extension FITS with PRED_MASK HDU: {segmap_fits_path}")

    print("\n" + "=" * 75)
    print(f"Tutorial 06 completed successfully!")
    print(f"Summary of Generated Products:")
    print(f"  1. 2-Panel Prediction Plot: {fig_path}")
    if len(res.masks) > 0:
        print(f"  2. Instance Overlay Plot:   {overlay_path}")
    print(f"  3. Multi-Extension FITS:    {segmap_fits_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
