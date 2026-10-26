#!/usr/bin/env python
"""
Tutorial 02: Training & Fine-Tuning SAM 2 on an Astronomical FITS Dataset
========================================================================

This script demonstrates how to train/fine-tune SAM 2 using:
1. Expected data formats (Single master FITS with tiling or multi-file datasets).
2. Creating an AstronomicalDataset with automatic prompt extraction (box or point).
3. Freezing the Hiera vision backbone to preserve pre-trained features and save memory.
4. Training the prompt encoder and mask decoder with Combined BCE + Dice Loss.
5. Evaluating validation loss and validation IoU per epoch.
6. Saving checkpoints and running inference with the fine-tuned model.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2roman.io.fits import FITSImage
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.trainer import SAM2Trainer
from sam2roman.inference.segmentor import SAM2Segmentor
from sam2roman.utils.synthetic import generate_synthetic_roman_fits
from sam2roman.visualization.display import plot_masks_overlay


def main():
    print("=" * 70)
    print("   SAM 2 for Roman Space Telescope: Tutorial 02 - Fine-Tuning Pipeline   ")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "sample_data"
    output_dir = base_dir / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_fits_path = data_dir / "sample_roman_train_field.fits"

    # Step 1: Create or Load Master Astronomical Image + Ground Truth SegMap
    print("\n[Step 1] Preparing astronomical dataset...")
    if not master_fits_path.exists():
        print(f"  -> Generating synthetic Roman WFI field with ground truth: {master_fits_path.name}")
        image_data, header, catalog, gt_segmap = generate_synthetic_roman_fits(
            output_path=master_fits_path,
            shape=(1024, 1024),
            n_stars=60,
            n_galaxies=35,
            seed=42,
            filter_name="F158",
        )
    else:
        fits_img = FITSImage.from_file(master_fits_path)
        image_data = fits_img.data
        # Generate synthetic segmap if loading existing file without SEGMAP
        _, _, catalog, gt_segmap = generate_synthetic_roman_fits(
            shape=(1024, 1024), n_stars=60, n_galaxies=35, seed=42
        )

    print(f"  -> Master field dimensions: {image_data.shape}")
    print(f"  -> Ground truth sources:    {len(catalog)} objects in field")

    # Step 2: Slice the single master FITS into tiled training cutouts
    print("\n[Step 2] Slicing single FITS into overlapping training cutouts (512x512)...")
    full_dataset = AstronomicalDataset.from_single_fits_tiled(
        fits_image=image_data,
        segmap=gt_segmap,
        tile_size=512,
        stride=256,
        min_objects=2,
        prompt_type="point",  # 'point' or 'box'
        stretch_type="asinh",
    )
    print(f"  -> Generated {len(full_dataset)} training tiles from 1 single master image!")

    # Step 3: Split into Train (80%) and Validation (20%)
    n_train = max(1, int(0.8 * len(full_dataset)))
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"  -> Training cutouts:   {len(train_dataset)}")
    print(f"  -> Validation cutouts: {len(val_dataset)}")

    # Step 4: Configure SAM 2 Trainer
    model_id = "facebook/sam2.1-hiera-tiny"
    print(f"\n[Step 4] Configuring SAM2Trainer with '{model_id}'...")
    trainer = SAM2Trainer(
        model_id=model_id,
        learning_rate=1e-4,       # Standard AdamW learning rate for mask decoder
        weight_decay=1e-4,        # L2 Regularization
        freeze_vision_encoder=True,  # FREEZE heavy Hiera backbone to preserve representations
        prompt_type="point",
    )
    print(f"  -> Execution Device:   {trainer.device}")
    print(f"  -> Trainable weights:  {sum(p.numel() for p in trainer.trainable_params):,} parameters")

    # Step 5: Execute Training Loop
    epochs = 4
    batch_size = 2
    print(f"\n[Step 5] Launching fine-tuning for {epochs} epochs (batch_size={batch_size})...")
    history = trainer.train(
        dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        verbose=True,
    )

    # Step 6: Plot Training & Validation Loss Curves
    loss_curve_path = base_dir / "02_training_curves.png"
    print(f"\n[Step 6] Saving training loss and IoU curves to: {loss_curve_path.name}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, epochs + 1)

    ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="royalblue")
    if history["val_loss"]:
        ax1.plot(epochs_range, history["val_loss"], "s--", label="Val Loss", color="crimson")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Combined Loss (BCE + Dice)")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if history["val_iou"]:
        ax2.plot(epochs_range, history["val_iou"], "^-", label="Val Mean IoU", color="forestgreen")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean IoU")
        ax2.set_title("Validation Segmentation Quality")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(loss_curve_path, dpi=200)
    plt.close(fig)
    print(f"  -> Saved loss plot to: {loss_curve_path}")

    # Step 7: Test Inference with Fine-Tuned Checkpoint
    best_ckpt = output_dir / "checkpoint_best.pt"
    if not best_ckpt.exists():
        best_ckpt = output_dir / f"checkpoint_epoch_{epochs}.pt"

    print(f"\n[Step 7] Testing inference with fine-tuned checkpoint: {best_ckpt.name}")
    val_sample = val_dataset[0]
    test_patch = val_sample["raw_data"]
    test_pt = val_sample["point"][0].tolist()

    # Load fine-tuned weights into segmentor
    fine_tuned_segmentor = SAM2Segmentor(model_id=model_id)
    fine_tuned_segmentor.model_wrapper.model.load_state_dict(
        torch.load(best_ckpt, map_location=trainer.device)["model_state_dict"]
    )
    fine_tuned_segmentor.model_wrapper.model.eval()

    res = fine_tuned_segmentor.segment_points(
        image=test_patch,
        points=[(test_pt[0], test_pt[1])],
        stretch_type="asinh",
        verbose=True,
    )
    print(f"  -> Successfully predicted mask on test patch! Area: {res.areas[0]} px, Score: {res.scores[0]:.3f}")

    print("\n" + "=" * 70)
    print("   Tutorial 02 Completed Successfully!   ")
    print("=" * 70)


if __name__ == "__main__":
    main()
