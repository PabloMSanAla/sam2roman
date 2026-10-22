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
