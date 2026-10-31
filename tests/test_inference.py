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


def test_asdf_image_mock(tmp_path):
    import warnings
    from sam2roman.io.asdf_io import ASDFImage, HAS_ASDF
    from sam2roman.io.reader import load_astro_image

    data = np.random.rand(64, 64).astype(np.float32) * 50.0
    meta = {"test": 123}

    img = ASDFImage(data=data, meta=meta)
    assert img.shape == (64, 64)
    rgb = img.to_rgb(stretch_type="leakysb")
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8

    pil_img = img.to_pil(stretch_type="leakysb")
    assert pil_img.size == (64, 64)

    if HAS_ASDF:
        import asdf
        asdf_path = tmp_path / "test_scene.asdf"
        tree = {
            "roman": {
                "data": data,
                "meta": meta,
            }
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            af = asdf.AsdfFile(tree)
            af.write_to(asdf_path)

        loaded = load_astro_image(asdf_path)
        assert isinstance(loaded, ASDFImage)
        assert loaded.shape == (64, 64)
        assert np.allclose(loaded.data, data)


def test_unetpp_predict_dense_direct_and_tiled():
    import torch
    from sam2roman.inference.unetpp_segmentor import UNetPPSegmentor

    # Lightweight model on CPU
    segmentor = UNetPPSegmentor(
        features=(8, 16, 32, 64, 128),
        device="cpu",
    )

    data = np.random.rand(128, 128).astype(np.float32) * 100.0

    # 1. Direct unchunked inference
    mask_dir, probs_dir, stretched_dir = segmentor.predict_dense(
        image=data,
        tile_size=0,
        stretch_type="leakysb",
        verbose=False,
    )
    assert mask_dir.shape == (128, 128)
    assert probs_dir.shape == (3, 128, 128)
    assert stretched_dir.shape == (128, 128)
    assert set(np.unique(mask_dir)).issubset({0, 1, 2})

    # 2. Tiled sliding window inference with cosine blending
    mask_tiled, probs_tiled, stretched_tiled = segmentor.predict_dense(
        image=data,
        tile_size=64,
        overlap=16,
        batch_size=2,
        stretch_type="leakysb",
        verbose=False,
    )
    assert mask_tiled.shape == (128, 128)
    assert probs_tiled.shape == (3, 128, 128)
    assert stretched_tiled.shape == (128, 128)
    assert set(np.unique(mask_tiled)).issubset({0, 1, 2})
    # Probabilities should sum to ~1.0 across classes
    prob_sums = np.sum(probs_tiled, axis=0)
    assert np.allclose(prob_sums, 1.0, atol=1e-4)


def test_two_panel_visualization(tmp_path):
    import matplotlib.pyplot as plt
    from sam2roman.visualization.display import plot_two_panel_prediction

    stretched = np.random.rand(64, 64).astype(np.float32)
    pred_mask = np.random.choice([0, 1, 2], size=(64, 64)).astype(np.uint8)

    save_file = tmp_path / "test_two_panel.png"
    fig = plot_two_panel_prediction(
        stretched=stretched,
        pred_mask=pred_mask,
        title="Test 2-Panel Figure",
        save_path=save_file,
        dpi=100,
    )
    assert fig is not None
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)

