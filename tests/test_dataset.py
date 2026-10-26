"""Unit tests for AstronomicalDataset and tiling."""

import numpy as np
from sam2roman.training.dataset import AstronomicalDataset


def test_astronomical_dataset_item():
    img = np.random.normal(10.0, 1.0, (128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[40:60, 40:60] = 1

    ds = AstronomicalDataset(images=[img], masks=[mask], prompt_type="point")
    assert len(ds) == 1
    sample = ds[0]

    assert "image" in sample
    assert "pil_image" in sample
    assert "mask" in sample
    assert "box" in sample
    assert "point" in sample
    assert sample["image"].shape == (3, 128, 128)
    assert sample["mask"].shape == (1, 128, 128)


def test_from_single_fits_tiled():
    img = np.random.normal(10.0, 1.0, (512, 512)).astype(np.float32)
    mask = np.zeros((512, 512), dtype=np.int32)
    # Add multiple objects
    for i in range(1, 10):
        rx, ry = 50 * i, 50 * i
        mask[ry:ry+10, rx:rx+10] = i

    ds = AstronomicalDataset.from_single_fits_tiled(
        fits_image=img,
        segmap=mask,
        tile_size=256,
        stride=128,
        min_objects=1,
    )

    assert len(ds) > 1
    sample = ds[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
