#!/usr/bin/env python
"""
Tutorial 03: Training SAM 2 Using Paired FITS Images and Masks Directories
==========================================================================

Demonstrates fine-tuning SAM 2 with:
- Images directory: /Volumes/T7/storage/ROMAN/test/synthetic/original/
- Masks directory:  /Volumes/T7/storage/ROMAN/test/synthetic/masks/
- Point prompts instead of boxes (foreground object clicks)
- Fast on-the-fly 512x512 cutout sampling from 4088x4088 detector exposures
- Small epoch test run with validation metrics and checkpointing
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.inference.segmentor import SAM2Segmentor


def main():
    print("=" * 75)
    print("   SAM 2 for Roman: Tutorial 03 - Training from Paired Directories   ")
    print("=" * 75)

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Target Directories
    images_dir = Path("/Volumes/T7/storage/ROMAN/test/synthetic/original")
    masks_dir = Path("/Volumes/T7/storage/ROMAN/test/synthetic/masks")

    # Fallback to sample_data if external drive is not mounted
    if not images_dir.exists() or not masks_dir.exists():
        print(f"\n[Warning] External path '{images_dir}' not accessible.")
        print("Falling back to local 'tutorials/sample_data/' directory.")
        images_dir = base_dir / "sample_data"
        masks_dir = base_dir / "sample_data"

    print(f"\n[Step 1] Loading image-mask pairs from:")
    print(f"  -> Images: {images_dir}")
    print(f"  -> Masks:  {masks_dir}")

    # 2. Build Dataset from Directories with Point Prompts
    # - all_masks_in_crop=True: segments ALL objects inside each 512x512 crop simultaneously!
    # - samples_per_file=None: extracts ALL patches possible (one cutout centered on every object)
    # - stretch_type="leakysb": default surface brightness transformation with identical RGB channels
    max_samples = 4  # Number of FITS files to match (or None for all 108 files)
    dataset = AstronomicalDataset.from_directories(
        images_dir=images_dir,
        masks_dir=masks_dir,
        pattern="*.fits",
        crop_size=512,
        prompt_type="point",       # POINT PROMPTS as requested
        samples_per_file=None,     # None = ALL patches possible across all objects!
        all_masks_in_crop=True,    # Trains on ALL masks present inside that cropped region!
        max_objects_per_crop=16,   # Cap to max 16 objects per crop to bound VRAM
        stretch_type="leakysb",    # Default: leakySB surface brightness stretch
        max_samples=max_samples,
    )
    print(f"  -> Extracted ALL {len(dataset)} possible multi-object patches from {max_samples} exposures!")

    # 3. Train / Validation Split (80% / 20%)
    n_train = max(1, int(0.8 * len(dataset)))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"  -> Training samples:   {len(train_ds)}")
    print(f"  -> Validation samples: {len(val_ds)}")

    # 4. Configure SAM 2 Trainer
    model_id = "facebook/sam2.1-hiera-tiny"
    print(f"\n[Step 2] Initializing SAM2Trainer (prompt_type='point')...")
    trainer = SAM2Trainer(
        model_id=model_id,
        learning_rate=1e-4,
        weight_decay=1e-4,
        freeze_vision_encoder=True,  # Keep backbone frozen for fast adaptation
        prompt_type="point",
    )
    print(f"  -> Accelerator:        {trainer.device}")
    print(f"  -> Trainable weights:  {sum(p.numel() for p in trainer.trainable_params):,} params")

    # 5. Train for Small Epochs (e.g. 3 epochs)
    epochs = 3
    batch_size = 2
    print(f"\n[Step 3] Training for {epochs} epochs (batch_size={batch_size})...")
    history = trainer.train(
        dataset=train_ds,
        val_dataset=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        verbose=True,
    )

    # 6. Plot Loss and Validation Curves
    curve_path = base_dir / "03_directory_training_curves.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, epochs + 1)

    ax1.plot(ep_range, history["train_loss"], "o-", label="Train Loss", color="royalblue")
    if history["val_loss"]:
        ax1.plot(ep_range, history["val_loss"], "s--", label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE + Dice)")
    ax1.set_title("Training Loss Convergence")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history["val_iou"]:
        ax2.plot(ep_range, history["val_iou"], "^-", label="Validation IoU", color="forestgreen")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean IoU")
        ax2.set_title("Validation Segmentation Metric")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    print(f"\n[Step 4] Saved training curves to: {curve_path.name}")

    # 7. Test Inference with Fine-Tuned Checkpoint
    best_ckpt = output_dir / "checkpoint_best.pt"
    if not best_ckpt.exists():
        best_ckpt = output_dir / f"checkpoint_epoch_{epochs}.pt"

    print(f"\n[Step 5] Testing inference with fine-tuned checkpoint: {best_ckpt.name}")
    val_sample = val_ds[0]
    test_patch = val_sample["raw_data"]
    test_pt = val_sample["point"][0].tolist()

    segmentor = SAM2Segmentor(model_id=model_id)
    segmentor.model_wrapper.model.load_state_dict(
        torch.load(best_ckpt, map_location=trainer.device)["model_state_dict"]
    )
    segmentor.model_wrapper.model.eval()

    res = segmentor.segment_points(
        image=test_patch,
        points=[(test_pt[0], test_pt[1])],
        stretch_type="leakysb",
        verbose=True,
    )

    if res.masks:
        print(f"  -> Prediction successful! Mask Area: {res.areas[0]} px, Score: {res.scores[0]:.3f}")

    print("\n" + "=" * 75)
    print("   Tutorial 03 Completed Successfully!   ")
    print("=" * 75)


if __name__ == "__main__":
    main()
