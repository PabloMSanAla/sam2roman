#!/usr/bin/env python
"""
Tutorial 05: Training Boundary-Aware UNet++ on Astronomical FITS Images
======================================================================

This script demonstrates how to:
1. Load paired FITS images and masks using `target_mode='dense_3class'`.
2. Generate 3-class boundary-aware targets:
   - Class 0: Sky background
   - Class 1: Astronomical source interior
   - Class 2: Source boundary / edge & inter-object contact interfaces
3. Automatically optimize for Apple Silicon MPS or NVIDIA CUDA GPUs (A100 / L4 / T4).
4. Train UNet++ end-to-end with Combined Cross-Entropy + Multi-Class Dice Loss.
5. Track validation loss, interior IoU, and edge IoU per epoch.
6. Run promptless inference with marker-controlled watershed deblending for touching objects.
7. Measure source photometry and export multi-extension FITS.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.unetpp_trainer import UNetPPTrainer
from sam2roman.inference.unetpp_segmentor import UNetPPSegmentor
from sam2roman.visualization.display import plot_detection_summary, plot_masks_overlay


def main():
    print("=" * 75)
    print("   UNet++ for Roman: Boundary-Aware 3-Class Training & Deblending   ")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. Hardware Detection & Device Optimization
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device = "cuda"
        batch_size = 16 if vram_gb >= 30.0 else 8
        num_workers = min(4, os.cpu_count() or 4)
        print(f"\n[Hardware] NVIDIA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  -> CUDA Optimizations: TF32=True, AMP (bfloat16), pin_memory=True")
    elif torch.backends.mps.is_available():
        device = "mps"
        batch_size = 4
        num_workers = 0
        print(f"\n[Hardware] Apple Silicon MPS (Metal Performance Shaders active)")
    else:
        device = "cpu"
        batch_size = 2
        num_workers = 0
        print(f"\n[Hardware] CPU fallback")

    # 1. Target Directories
    images_dir = Path("/Volumes/T7/storage/ROMAN/test/synthetic/original")
    masks_dir = Path("/Volumes/T7/storage/ROMAN/test/synthetic/masks")

    if not images_dir.exists() or not masks_dir.exists():
        print(f"\n[Warning] External paths not accessible. Using local 'sample_data' fallback.")
        images_dir = base_dir / "sample_data"
        masks_dir = base_dir / "sample_data"

    print(f"\n[Step 1] Loading image-mask pairs from:")
    print(f"  -> Images: {images_dir}")
    print(f"  -> Masks:  {masks_dir}")

    # 2. Build Dataset with 3-Class Boundary Targets
    dataset = AstronomicalDataset.from_directories(
        images_dir=images_dir,
        masks_dir=masks_dir,
        pattern="*.fits",
        crop_size=512,
        samples_per_file=8,          # 8 cutouts per file for quick demonstration (None for all)
        target_mode="dense_3class",  # Returns (3, H, W) image and (H, W) 3-class target {0, 1, 2}
        stretch_type="leakysb",      # Default surface brightness stretch
        max_samples=2,               # 2 exposures for quick demonstration
    )
    print(f"  -> Extracted {len(dataset)} training cutouts with boundary targets!")

    # 3. Train / Validation Split (80% / 20%)
    n_train = max(1, int(0.8 * len(dataset)))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"  -> Training cutouts:   {len(train_ds)}")
    print(f"  -> Validation cutouts: {len(val_ds)}")

    # 4. Initialize UNet++ Trainer
    print(f"\n[Step 2] Initializing UNetPPTrainer...")
    trainer = UNetPPTrainer(
        in_channels=3,
        num_classes=3,               # 0: Sky, 1: Interior, 2: Boundary/Edge
        features=(32, 64, 128, 256, 512),
        learning_rate=3e-4,
        class_weights=(1.0, 2.5, 4.0),  # Weights emphasizing interior and edge
        device=device,
    )

    # 5. Train for 5 Epochs
    epochs = 5
    print(f"\n[Step 3] Training UNet++ for {epochs} epochs (batch_size={batch_size})...")
    history = trainer.train(
        dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        verbose=True,
    )

    # 6. Plot Loss and Validation Metric Curves
    curve_path = base_dir / "05_unetpp_training_curves.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, epochs + 1)

    ax1.plot(ep_range, history["train_loss"], "o-", label="Train Loss (CE + Dice)", color="royalblue")
    if history["val_loss"]:
        ax1.plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("UNet++ Training Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history["val_macro_iou"]:
        ax2.plot(ep_range, history["val_macro_iou"], "^-", label="Macro IoU", color="purple")
        ax2.plot(ep_range, history["val_interior_iou"], "o--", label="Interior IoU", color="forestgreen")
        ax2.plot(ep_range, history["val_boundary_iou"], "d--", label="Edge IoU", color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IoU Metric")
        ax2.set_title("Validation Segmentation Metrics")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"\n[Step 4] Saved UNet++ training curves to: {curve_path.name}")

    # 7. Test Inference & Instance Deblending with Trained Checkpoint
    best_ckpt = output_dir / "unetpp_best.pt"
    if not best_ckpt.exists():
        best_ckpt = output_dir / f"unetpp_epoch_{epochs}.pt"

    print(f"\n[Step 5] Running inference with fine-tuned UNet++: {best_ckpt.name}")
    segmentor = UNetPPSegmentor.from_checkpoint(best_ckpt, device=device)

    # Test on a validation cutout
    val_sample = val_ds[0]
    test_raw = val_sample["raw_data"]

    results = segmentor.segment_image(
        image=test_raw,
        stretch_type="leakysb",
        interior_thresh=0.45,
        min_area=10,
        verbose=True,
    )
    print(f"  -> Detected {len(results)} discrete deblended astronomical sources!")

    # 8. Measure Source Photometry
    print(f"\n[Step 6] Measuring source photometry on deblended masks...")
    catalog = results.compute_photometry(test_raw, verbose=True)
    if catalog:
        brightest = max(catalog, key=lambda s: s["net_flux"])
        print(f"  -> Brightest source: ID {brightest['source_id']}, Flux: {brightest['net_flux']:.1f}, SNR: {brightest['snr']:.1f}")

    # 9. Plot 3-Panel Detection Summary
    summary_path = base_dir / "05_unetpp_detection_summary.png"
    fig_summary = plot_detection_summary(
        image=test_raw,
        seg_result=results,
        title_prefix="UNet++ Segmented Cutout",
        save_path=summary_path,
    )
    plt.close(fig_summary)
    print(f"  -> Saved detection summary to: {summary_path.name}")

    print("\n" + "=" * 75)
    print("   Tutorial 05 Completed Successfully!   ")
    print("=" * 75)


if __name__ == "__main__":
    main()
