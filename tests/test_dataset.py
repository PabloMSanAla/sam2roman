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


def test_multi_object_crop():
    img = np.random.normal(10.0, 1.0, (256, 256)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.int32)
    # Add 4 discrete objects
    for i in range(1, 5):
        rx, ry = 40 * i, 40 * i
        mask[ry : ry + 12, rx : rx + 12] = i

    ds = AstronomicalDataset(
        images=[img],
        masks=[mask],
        all_masks_in_crop=True,
    )
    sample = ds[0]
    assert sample["num_objects"] == 4
    assert sample["masks"].shape == (4, 1, 256, 256)
    assert sample["points"].shape == (4, 1, 2)
    assert sample["boxes"].shape == (4, 4)
    assert sample["point_labels"].shape == (4, 1)
    # Compatibility
    assert sample["mask"].shape == (1, 256, 256)


def test_single_object_mode():
    img = np.random.normal(10.0, 1.0, (256, 256)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.int32)
    for i in range(1, 5):
        rx, ry = 40 * i, 40 * i
        mask[ry : ry + 12, rx : rx + 12] = i

    ds = AstronomicalDataset(
        images=[img],
        masks=[mask],
        all_masks_in_crop=False,
    )
    sample = ds[0]
    assert sample["num_objects"] == 1
    assert sample["masks"].shape == (1, 1, 256, 256)


def test_max_objects_cap():
    img = np.random.normal(10.0, 1.0, (256, 256)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=np.int32)
    for i in range(1, 10):
        rx, ry = 25 * i, 25 * i
        mask[ry : ry + 10, rx : rx + 10] = i

    ds = AstronomicalDataset(
        images=[img],
        masks=[mask],
        all_masks_in_crop=True,
        max_objects_per_crop=3,
    )
    sample = ds[0]
    assert sample["num_objects"] == 3
    assert sample["masks"].shape == (3, 1, 256, 256)


def test_leakysb_default_and_rgb_identical_channels():
    img = np.random.normal(15.0, 2.0, (128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=np.int32)
    mask[30:50, 30:50] = 1

    ds = AstronomicalDataset(images=[img], masks=[mask])
    assert ds.stretch_type == "leakysb"

    sample = ds[0]
    # Check shape: 3 channels
    image_tensor = sample["image"]
    assert image_tensor.shape == (3, 128, 128)

    # Check identical channels for R, G, B
    np.testing.assert_allclose(image_tensor[0].numpy(), image_tensor[1].numpy(), atol=1e-6)
    np.testing.assert_allclose(image_tensor[1].numpy(), image_tensor[2].numpy(), atol=1e-6)

    # Check PIL image has 3 identical channels
    pil_arr = np.array(sample["pil_image"])
    assert pil_arr.shape == (128, 128, 3)
    np.testing.assert_array_equal(pil_arr[..., 0], pil_arr[..., 1])
    np.testing.assert_array_equal(pil_arr[..., 1], pil_arr[..., 2])


def test_centered_point_prompt():
    img = np.ones((100, 100), dtype=np.float32) * 5.0
    mask = np.zeros((100, 100), dtype=np.int32)
    # Square object from y: 30..50, x: 40..60
    # Mean y = 39.5, Mean x = 49.5
    mask[30:50, 40:60] = 1

    ds = AstronomicalDataset(
        images=[img],
        masks=[mask],
        point_prompt_mode="center",
    )
    sample = ds[0]
    pt = sample["point"][0].tolist()
    assert abs(pt[0] - 49.5) < 1e-4
    assert abs(pt[1] - 39.5) < 1e-4

    # Verify for concave/ring mask where centroid is in empty space
    ring_mask = np.zeros((100, 100), dtype=np.int32)
    ring_mask[40:60, 40:60] = 1
    ring_mask[45:55, 45:55] = 0  # Hole in the middle!
    ds_ring = AstronomicalDataset(
        images=[img],
        masks=[ring_mask],
        point_prompt_mode="center",
    )
    sample_ring = ds_ring[0]
    pt_ring = sample_ring["point"][0].tolist()
    # Ensure point is inside the actual ring mask (not in the hole)
    px, py = int(round(pt_ring[0])), int(round(pt_ring[1]))
    assert ring_mask[py, px] == 1

