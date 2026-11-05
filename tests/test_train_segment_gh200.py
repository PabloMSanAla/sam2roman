"""
Unit tests for Grace Hopper (GH200) SteerQuery training script:
- CLI argument parsing
- SteerQueryTrainer epoch training, teacher-forcing, and checkpoint saving
- Plotting functions for curves and 5-column validation figures
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from sam2roman.models.steer_query_head import UNetPlusPlusSteerQuery
from sam2roman.training.steer_query_trainer import SteerQueryTrainer
from cloud.train_segment_gh200 import (
    build_parser,
    extract_fixed_val_cutouts,
    extract_crowded_val_cutouts,
    select_val_cutouts_for_epoch,
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
        "--sem_weight", "1.0",
        "--center_weight", "2.5",
        "--gate_weight", "1.2",
        "--inst_weight", "1.5",
    ])
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.num_workers == 8
    assert args.prefetch_factor == 4
    assert args.sem_weight == 1.0
    assert args.center_weight == 2.5
    assert args.gate_weight == 1.2
    assert args.inst_weight == 1.5
    assert args.rot_prob == 0.0


def test_steer_query_trainer_run(tmp_path):
    """Verify SteerQueryTrainer trains for 2 epochs and saves checkpoints."""
    ds = DummyInstanceDataset(n_samples=4)
    loader = DataLoader(ds, batch_size=2)

    trainer = SteerQueryTrainer(
        in_channels=1,
        num_classes=3,
        features=(8, 16, 32, 64, 128),
        proto_dim=16,
        N_max=8,
        learning_rate=1e-3,
        gpu_transform=None,
        device="cpu",
    )

    history = trainer.train(
        dataset=loader,
        val_dataset=loader,
        epochs=2,
        batch_size=2,
        warmup_epochs=1,
        output_dir=tmp_path,
        log_file="epoch_log.txt",
        verbose=False,
    )

    assert len(history["loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert (tmp_path / "epoch_log.txt").exists()
    assert (tmp_path / "steer_query_latest.pt").exists()
    assert (tmp_path / "steer_query_best.pt").exists()
    best_ckpt = torch.load(tmp_path / "steer_query_best.pt", weights_only=False)
    assert "optimizer_state_dict" in best_ckpt
    assert (tmp_path / "training_metrics.json").exists()


def test_grad_accum_steps(tmp_path):
    """Verify training with gradient accumulation steps."""
    ds = DummyInstanceDataset(n_samples=4)
    loader = DataLoader(ds, batch_size=2)

    trainer = SteerQueryTrainer(
        in_channels=1,
        num_classes=3,
        features=(8, 16, 32, 64, 128),
        proto_dim=16,
        N_max=8,
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
        "gate_loss": [0.2, 0.15],
        "inst_loss": [0.1, 0.05],
        "val_loss": [0.7, 0.45],
        "fg_iou": [0.5, 0.7],
        "star_f1": [0.6, 0.8],
    }

    curve_path = plot_training_curves(history, tmp_path, current_epoch=2)
    assert curve_path.exists()

    model = UNetPlusPlusSteerQuery(
        in_channels=1,
        num_classes=3,
        features=(8, 16, 32, 64, 128),
        proto_dim=16,
        N_max=8,
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


def test_select_crowded_cutouts_rotation():
    """Verify random sampling from the most crowded cutouts across epochs."""
    # Pool of 30 cutouts sorted descending by stars (ids 0..9 are most crowded)
    pool = [{"id": i, "n_stars": 100 - i * 3} for i in range(30)]

    c_ep5 = select_val_cutouts_for_epoch(pool, epoch=5, num_cutouts=4, crowded_top_k=10)
    c_ep10 = select_val_cutouts_for_epoch(pool, epoch=10, num_cutouts=4, crowded_top_k=10)
    c_ep15 = select_val_cutouts_for_epoch(pool, epoch=15, num_cutouts=4, crowded_top_k=10)

    # 1. All selected cutouts must come exclusively from the top-10 crowded set
    for batch in [c_ep5, c_ep10, c_ep15]:
        assert len(batch) == 4
        for item in batch:
            assert item["id"] < 10

    # 2. Different epochs choose different combinations
    ids_ep5 = set(c["id"] for c in c_ep5)
    ids_ep10 = set(c["id"] for c in c_ep10)
    assert ids_ep5 != ids_ep10

    # 3. Deterministic for same epoch
    c_ep5_repeat = select_val_cutouts_for_epoch(pool, epoch=5, num_cutouts=4, crowded_top_k=10)
    assert [c["id"] for c in c_ep5] == [c["id"] for c in c_ep5_repeat]
