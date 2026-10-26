"""Unit tests for SAM2Trainer with multi-mask and single-mask crops."""

import numpy as np
import pytest
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.trainer import SAM2Trainer


def test_sam2_trainer_multi_object():
    img1 = np.random.normal(10.0, 1.0, (128, 128)).astype(np.float32)
    mask1 = np.zeros((128, 128), dtype=np.int32)
    mask1[20:40, 20:40] = 1
    mask1[60:80, 60:80] = 2

    img2 = np.random.normal(10.0, 1.0, (128, 128)).astype(np.float32)
    mask2 = np.zeros((128, 128), dtype=np.int32)
    mask2[30:50, 30:50] = 1

    ds = AstronomicalDataset(images=[img1, img2], masks=[mask1, mask2], all_masks_in_crop=True)
    val_ds = AstronomicalDataset(images=[img1], masks=[mask1], all_masks_in_crop=True)

    trainer = SAM2Trainer(device="cpu", prompt_type="point")
    history = trainer.train(ds, val_dataset=val_ds, epochs=1, batch_size=2, verbose=False)

    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0
    assert len(history["val_loss"]) == 1
    assert history["val_iou"][0] >= 0.0


def test_sam2_trainer_box_mode():
    img = np.random.normal(10.0, 1.0, (128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[20:40, 20:40] = 1
    mask[60:80, 60:80] = 2

    ds = AstronomicalDataset(images=[img], masks=[mask], all_masks_in_crop=True)
    trainer = SAM2Trainer(device="cpu", prompt_type="box")
    history = trainer.train(ds, epochs=1, batch_size=1, verbose=False)

    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0
