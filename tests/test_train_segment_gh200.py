"""
Unit tests for Grace Hopper (GH200) Center Offset training script:
- CLI argument parsing
- CenterOffsetTrainer epoch training and checkpoint saving
- Plotting functions for curves and 5-column validation figures
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from sam2roman.models.unetpp_offset import UNetPlusPlusCenterOffset
from sam2roman.training.center_offset_trainer import CenterOffsetTrainer
from cloud.train_segment_gh200 import (
    build_parser,
    extract_fixed_val_cutouts,
    plot_training_curves,
    plot_validation_examples,
)


class DummyInstanceDataset(Dataset):
    def __init__(self, n_samples: int = 6):
        self.samples = []
        for i in range(n_samples):
            raw = np.random.normal(10.0, 1.0, (64, 64)).astype(np.float32)
            mask = np.zeros((64, 64), dtype=np.int32)
            # Two distinct instances
            mask[10:20, 10:20] = 1
            mask[35:45, 35:45] = 2
            self.samples.append({"raw": raw, "target": mask, "meta": {}})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "raw": torch.from_numpy(item["raw"]).unsqueeze(0),
            "target": torch.from_numpy(item["target"]),
            "meta": item["meta"],
        }


def test_cli_parser():
    """Verify build_parser arguments."""
    parser = build_parser()
    args = parser.parse_args([
        "--train_shards", "shards/train/*.tar",
        "--val_shards", "shards/val/*.tar",
        "--output_dir", "/tmp/test_gh200",
        "--epochs", "10",
        "--batch_size", "16",
        "--num_workers", "8",
        "--prefetch_factor", "4",
        "--center_weight", "2.5",
        "--offset_weight", "1.5",
        "--angle_weight", "0.8",
    ])
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.num_workers == 8
    assert args.prefetch_factor == 4
    assert args.center_weight == 2.5
    assert args.offset_weight == 1.5
    assert args.angle_weight == 0.8
    assert args.use_cca is True


def test_center_offset_trainer_run(tmp_path):
    """Verify CenterOffsetTrainer trains for 2 epochs and saves checkpoints."""
    ds = DummyInstanceDataset(n_samples=4)
    loader = DataLoader(ds, batch_size=2)

    trainer = CenterOffsetTrainer(
        in_channels=1,  # Raw 1-channel for dummy test
        features=(8, 16, 32, 64, 128),
        use_cca=False,
        learning_rate=1e-3,
        gpu_transform=None,
        device="cpu",
    )

    history = trainer.train(
        dataset=loader,
        val_dataset=loader,
        epochs=2,
        batch_size=2,
        output_dir=tmp_path,
        log_file="epoch_log.txt",
        verbose=False,
    )

    assert len(history["loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert (tmp_path / "epoch_log.txt").exists()
    assert (tmp_path / "center_offset_latest.pt").exists()
    assert (tmp_path / "center_offset_best.pt").exists()
    best_ckpt = torch.load(tmp_path / "center_offset_best.pt", weights_only=False)
    assert "optimizer_state_dict" in best_ckpt
    assert (tmp_path / "training_metrics.json").exists()


def test_center_offset_batch_collator():
    """Verify CenterOffsetBatchCollator precomputes fg, center, and offset asynchronously."""
    from sam2roman.training.webdataset_loader import CenterOffsetBatchCollator

    collator = CenterOffsetBatchCollator(sigma=2.0)
    raw_1 = np.ones((64, 64), dtype=np.float32)
    tgt_1 = np.zeros((64, 64), dtype=np.int32)
    tgt_1[20:30, 20:30] = 1

    batch_input = [
        (raw_1, tgt_1, {"id": 1}),
        (raw_1 * 2, tgt_1 * 2, {"id": 2}),
    ]
    collated = collator(batch_input)

    assert "raw" in collated
    assert "target" in collated
    assert "fg" in collated
    assert "center" in collated
    assert "offset" in collated
    assert collated["fg"].shape == (2, 1, 64, 64)
    assert collated["center"].shape == (2, 1, 64, 64)
    assert collated["offset"].shape == (2, 2, 64, 64)


def test_grad_accum_steps(tmp_path):
    """Verify training with gradient accumulation steps."""
    ds = DummyInstanceDataset(n_samples=4)
    loader = DataLoader(ds, batch_size=2)

    trainer = CenterOffsetTrainer(
        in_channels=1,
        features=(8, 16, 32, 64, 128),
        use_cca=False,
        learning_rate=1e-3,
        gpu_transform=None,
        device="cpu",
    )

    history = trainer.train(
        dataset=loader,
        val_dataset=loader,
        epochs=1,
        batch_size=2,
        grad_accum_steps=2,
        output_dir=tmp_path / "grad_accum",
        verbose=False,
    )
    assert len(history["loss"]) == 1


def test_plot_curves_and_validation(tmp_path):
    """Verify plotting functions execute and write image files."""
    history = {
        "loss": [0.8, 0.5],
        "sem_loss": [0.4, 0.2],
        "center_loss": [0.2, 0.15],
        "offset_loss": [0.2, 0.15],
        "angle_loss": [0.1, 0.05],
        "val_loss": [0.7, 0.45],
        "fg_iou": [0.5, 0.7],
        "center_f1": [0.6, 0.8],
        "offset_epe": [3.5, 2.1],
    }

    curve_path = plot_training_curves(history, tmp_path, current_epoch=2)
    assert curve_path.exists()

    model = UNetPlusPlusCenterOffset(
        in_channels=1,
        features=(8, 16, 32, 64, 128),
        use_cca=False,
    )

    cutouts = [
        {
            "raw": np.random.rand(64, 64).astype(np.float32),
            "target": np.zeros((64, 64), dtype=np.int32),
            "meta": {},
        }
    ]
    cutouts[0]["target"][15:25, 15:25] = 1

    fig_path = tmp_path / "val_inference_epoch_002.png"
    out = plot_validation_examples(
        model=model,
        fixed_cutouts=cutouts,
        gpu_transform=None,
        output_path=fig_path,
        epoch=2,
        epochs=2,
        device="cpu",
    )
    assert out.exists()
