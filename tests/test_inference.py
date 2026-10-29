"""Unit tests for SegmentationResult and inference logic."""

import numpy as np
from sam2roman.inference.segmentor import SegmentationResult


def test_segmentation_result_filtering():
    m1 = np.zeros((32, 32), dtype=bool)
    m1[0:2, 0:2] = True  # area = 4
    m2 = np.zeros((32, 32), dtype=bool)
    m2[10:20, 10:20] = True  # area = 100

    res = SegmentationResult(
        masks=[m1, m2],
        boxes=[[0, 0, 2, 2], [10, 10, 20, 20]],
        scores=[0.6, 0.95],
        areas=[4, 100],
        centroids=[(1.0, 1.0), (15.0, 15.0)],
        image_shape=(32, 32),
    )

    assert len(res) == 2

    # Filter by area (e.g. min area 10)
    filtered = res.filter_by_area(min_area=10)
    assert len(filtered) == 1
    assert filtered.areas[0] == 100

    # Filter by score
    scored = res.filter_by_score(min_score=0.8)
    assert len(scored) == 1
    assert scored.scores[0] == 0.95

    # Combined integer mask
    comb = res.to_combined_mask()
    assert comb.shape == (32, 32)
    assert np.max(comb) == 2


def test_segmentor_from_checkpoint(tmp_path):
    from sam2roman.inference.segmentor import SAM2Segmentor
    import torch

    # Create dummy checkpoint
    ckpt_file = tmp_path / "dummy_checkpoint.pt"
    # Create a small dummy state_dict
    segmentor_init = SAM2Segmentor(device="cpu")
    torch.save(
        {
            "model_state_dict": segmentor_init.model_wrapper.model.state_dict(),
            "model_id": "facebook/sam2.1-hiera-tiny",
        },
        ckpt_file,
    )

    # Load using classmethod
    seg_from_ckpt = SAM2Segmentor.from_checkpoint(ckpt_file, device="cpu")
    assert seg_from_ckpt.model_wrapper is not None
    assert seg_from_ckpt.model_wrapper.model is not None

    # Load using __init__ checkpoint_path
    seg_init_ckpt = SAM2Segmentor(checkpoint_path=ckpt_file, device="cpu")
    assert seg_init_ckpt.model_wrapper is not None

