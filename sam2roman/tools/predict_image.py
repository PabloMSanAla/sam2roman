#!/usr/bin/env python3
"""
Full-Image Astronomical Segmentation and LeakySB Visualization Tool.

Given an astronomical image in FITS (.fits, .fits.gz) or ASDF (.asdf, such as Roman L2 data),
this tool applies the LeakySB surface brightness stretch, executes boundary-aware UNet++
inference across the entire image (using seamless sliding-window tiling for large exposures),
and renders a high-contrast 2-panel visualization:
    Panel 1: LeakySB Stretched Astronomical Image
    Panel 2: Boundary-Aware 3-Class Predicted Segmentation Mask (Sky, Core, Edge)

Usage Examples:
    # Basic prediction on Roman L2 ASDF image
    python sam2roman/tools/predict_image.py image.asdf --checkpoint experiments/unet/test_1/unetpp_best.pt

    # Prediction on FITS image with saving
    python sam2roman/tools/predict_image.py scene.fits --save output.png --no_display

    # Using custom tile size and overlap
    python sam2roman/tools/predict_image.py large_scene.fits --tile_size 1024 --overlap 64 --batch_size 4
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
import numpy as np
import torch

from sam2roman.io.reader import load_astro_image
from sam2roman.inference.unetpp_segmentor import UNetPPSegmentor
from sam2roman.visualization.display import plot_two_panel_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run boundary-aware UNet++ inference on full FITS or Roman ASDF images and display leakySB & predicted mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the astronomical image file (.fits, .fits.gz, or .asdf).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/unet/test_1/unetpp_best.pt",
        help="Path to fine-tuned UNet++ model checkpoint (.pt).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Sliding tile size in pixels (must be divisible by 16). Set to 0 for unchunked direct inference.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in pixels between adjacent tiles for smooth window blending.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of tiles processed concurrently in GPU forward batches.",
    )
    parser.add_argument(
        "--stretch",
        type=str,
        default="leakysb",
        choices=["leakysb", "asinh", "zscale", "linear", "log", "sqrt"],
        help="Astronomical surface brightness stretch algorithm.",
    )
    parser.add_argument(
        "--zero_points",
        type=float,
        default=31.4,
        help="Photometric zero point for leakySB stretch (Roman WFI default: 31.4).",
    )
    parser.add_argument(
        "--pixel_scale",
        type=float,
        default=0.11,
        help="Plate scale in arcseconds/pixel (Roman WFI default: 0.11).",
    )
    parser.add_argument(
        "--mmin",
        type=float,
        default=13.0,
        help="Bright-end surface brightness magnitude limit for leakySB.",
    )
    parser.add_argument(
        "--mmax",
        type=float,
        default=32.0,
        help="Faint-end surface brightness magnitude limit for leakySB.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device for neural network inference.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional destination path to save the 2-panel figure (.png, .pdf, .jpg).",
    )
    parser.add_argument(
        "--save_mask",
        type=str,
        default=None,
        help="Optional destination path to save the raw predicted mask array (.fits, .npy).",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Headless execution mode: suppress interactive matplotlib GUI window.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="16,8",
        help="Figure dimensions in inches formatted as 'width,height'.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution for rendered and saved figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[sam2roman] ERROR: Input image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[sam2roman] ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve compute device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[sam2roman] Loading astronomical image: {image_path.name}", flush=True)
    t_load = time.time()
    astro_obj = load_astro_image(image_path)
    h, w = astro_obj.shape
    file_type = "ASDF (Roman L2)" if image_path.name.lower().endswith(".asdf") else "FITS"
    print(f"[sam2roman] Loaded {file_type} image: shape={w}x{h} px, dtype={astro_obj.data.dtype} in {time.time()-t_load:.2f}s", flush=True)

    # Load UNet++ model from checkpoint
    print(f"[sam2roman] Loading UNet++ checkpoint: {checkpoint_path} on device [{device}]...", flush=True)
    segmentor = UNetPPSegmentor.from_checkpoint(checkpoint_path, device=device)

    # Parse figure size
    try:
        fw, fh = map(float, args.figsize.split(","))
        figsize = (fw, fh)
    except Exception:
        figsize = (16, 8)

    # Execute dense inference across the whole image
    t_inf = time.time()
    pred_mask, probs, stretched = segmentor.predict_dense(
        image=astro_obj,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        stretch_type=args.stretch,
        zero_points=args.zero_points,
        pixel_scale=args.pixel_scale,
        mmin=args.mmin,
        mmax=args.mmax,
        verbose=True,
    )
    inf_duration = time.time() - t_inf

    # Compute summary statistics
    total_px = h * w
    n_sky = int(np.sum(pred_mask == 0))
    n_core = int(np.sum(pred_mask == 1))
    n_edge = int(np.sum(pred_mask == 2))

    print("-" * 65, flush=True)
    print(f"[sam2roman] Full-Image Prediction Summary ({w}x{h} = {total_px:,} pixels):", flush=True)
    print(f"  * Sky Background (0) : {n_sky:,} px ({100.0 * n_sky / total_px:.2f}%)", flush=True)
    print(f"  * Source Interior (1): {n_core:,} px ({100.0 * n_core / total_px:.2f}%)", flush=True)
    print(f"  * Deblending Edge (2): {n_edge:,} px ({100.0 * n_edge / total_px:.2f}%)", flush=True)
    print(f"  * Total Time         : {inf_duration:.2f}s", flush=True)
    print("-" * 65, flush=True)

    # Save raw mask array if requested
    if args.save_mask:
        out_mask_path = Path(args.save_mask)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        if out_mask_path.suffix.lower() == ".npy":
            np.save(out_mask_path, pred_mask)
            print(f"[sam2roman] Saved prediction mask to NPY: {out_mask_path}")
        elif out_mask_path.suffix.lower() in (".fits", ".fit"):
            from astropy.io import fits
            hdu = fits.PrimaryHDU(data=pred_mask)
            hdu.header["CREATOR"] = "sam2roman UNet++"
            hdu.header["SRCFILE"] = image_path.name
            hdu.writeto(out_mask_path, overwrite=True)
            print(f"[sam2roman] Saved prediction mask to FITS: {out_mask_path}")
        else:
            np.save(out_mask_path.with_suffix(".npy"), pred_mask)
            print(f"[sam2roman] Saved prediction mask to NPY: {out_mask_path.with_suffix('.npy')}")

    # Build 2-panel visualization
    fig_title = f"Roman Space Telescope Astronomical Segmentation: {image_path.name}"
    image_panel_title = f"{file_type} Image ({args.stretch.upper()} Stretch)"
    mask_panel_title = f"UNet++ 3-Class Prediction ({inf_duration:.2f}s on {device})"

    fig = plot_two_panel_prediction(
        stretched=stretched,
        pred_mask=pred_mask,
        title=fig_title,
        save_path=args.save,
        figsize=figsize,
        dpi=args.dpi,
        image_title=image_panel_title,
        mask_title=mask_panel_title,
    )

    if args.save:
        print(f"[sam2roman] Saved 2-panel figure to: {args.save}")

    if not args.no_display:
        try:
            plt.show()
        except Exception as e:
            print(f"[sam2roman] GUI display failed ({e}). Run with --save to write to disk.")


if __name__ == "__main__":
    main()
