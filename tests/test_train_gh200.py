"""
Unit tests for Grace Hopper (GH200) training script features:
- Epoch log file output in output directory
- Periodic loss and validation plotting every 5 epochs
- Fixed 4-cutout validation inference figure generation every 5 epochs
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from sam2roman.models.unetpp import UNetPlusPlus
from sam2roman.training.dataset import AstronomicalDataset
from sam2roman.training.unetpp_trainer import UNetPPTrainer
from cloud.train_gh200 import (
    extract_fixed_val_cutouts,
    plot_training_curves,
    plot_validation_examples,
)


@pytest.fixture
def dummy_dataset():
    imgs = [np.random.normal(10.0, 1.0, (64, 64)).astype(np.float32) for _ in range(6)]
    masks = [np.zeros((64, 64), dtype=np.int32) for _ in range(6)]
    for i, m in enumerate(masks):
        m[15 + i : 30 + i, 15 + i : 30 + i] = 1
        m[35:50, 35:50] = 2

    return AstronomicalDataset(
        images=imgs,
        masks=masks,
        target_mode="dense_3class",
        stretch_type="leakysb",
    )


def test_unetpp_epoch_log_and_callbacks(tmp_path, dummy_dataset):
    """Verify UNetPPTrainer writes epoch_log.txt and updates training_metrics.json."""
    trainer = UNetPPTrainer(
        features=(8, 16, 32, 64, 128),
        learning_rate=1e-4,
        device="cpu",
    )

    cb_calls = []

    def dummy_callback(trainer_inst, epoch_data, history):
        cb_calls.append(epoch_data["epoch"])

    epochs = 3
    history = trainer.train(
        dataset=dummy_dataset,
        val_dataset=dummy_dataset,
        epochs=epochs,
        batch_size=2,
        output_dir=tmp_path,
        log_file="epoch_log.txt",
        epoch_callback=dummy_callback,
        verbose=False,
    )

    # Verify callback calls
    assert cb_calls == [1, 2, 3]

    # Verify epoch_log.txt contents
    log_file = tmp_path / "epoch_log.txt"
    assert log_file.exists()
    content = log_file.read_text()
    assert "sam2roman UNet++ Training Log" in content
    assert "Epoch [01/03]" in content
    assert "Epoch [02/03]" in content
    assert "Epoch [03/03]" in content
    assert "Completed:" in content

    # Verify training_metrics.json
    metrics_file = tmp_path / "training_metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    assert len(metrics["train_loss"]) == 3
    assert len(metrics["val_loss"]) == 3


def test_extract_fixed_val_cutouts(dummy_dataset):
    """Verify extraction of fixed validation cutouts from dataset and dataloader."""
    # From Dataset
    cutouts = extract_fixed_val_cutouts(dummy_dataset, num_cutouts=4)
    assert len(cutouts) == 4
    for c in cutouts:
        assert "raw" in c
        assert "target" in c
        assert isinstance(c["raw"], np.ndarray)
        assert isinstance(c["target"], np.ndarray)

    # From DataLoader
    loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)
    cutouts_loader = extract_fixed_val_cutouts(loader, num_cutouts=4)
    assert len(cutouts_loader) == 4
    assert cutouts_loader[0]["raw"].shape == (64, 64)


def test_plot_training_curves(tmp_path):
    """Verify training curves are generated and saved correctly."""
    history = {
        "train_loss": [1.2, 1.0, 0.8, 0.7, 0.6],
        "val_loss": [1.1, 0.95, 0.85, 0.72, 0.65],
        "val_macro_iou": [0.2, 0.35, 0.45, 0.52, 0.58],
        "val_interior_iou": [0.3, 0.5, 0.65, 0.72, 0.78],
        "val_boundary_iou": [0.1, 0.2, 0.25, 0.32, 0.38],
        "val_strict_boundary_iou": [0.05, 0.1, 0.15, 0.18, 0.22],
        "val_soft_boundary_iou": [0.08, 0.15, 0.2, 0.24, 0.28],
    }

    out_file = plot_training_curves(
        history=history,
        output_dir=tmp_path,
        current_epoch=5,
        boundary_tolerance=1,
    )
    assert out_file.exists()
    assert (tmp_path / "gh200_training_curves.png").exists()
    assert (tmp_path / "training_curves_epoch_005.png").exists()

    # Verify PNG is valid
    img = Image.open(out_file)
    assert img.size[0] > 0 and img.size[1] > 0


def test_plot_validation_examples(tmp_path, dummy_dataset):
    """Verify 4-cutout grid figure generation."""
    model = UNetPlusPlus(
        in_channels=3,
        num_classes=3,
        features=(8, 16, 32, 64, 128),
    )
    cutouts = extract_fixed_val_cutouts(dummy_dataset, num_cutouts=4)
    assert len(cutouts) == 4

    out_path = tmp_path / "val_inference_epoch_005.png"
    result = plot_validation_examples(
        model=model,
        fixed_cutouts=cutouts,
        gpu_transform=None,
        output_path=out_path,
        epoch=5,
        epochs=10,
        device="cpu",
    )
    assert result.exists()
    assert (tmp_path / "val_inference_latest.png").exists()

    # Check generated image dimensions
    img = Image.open(result)
    assert img.size[0] > 500
    assert img.size[1] > 500


def test_gh200_training_loop_5_epochs(tmp_path, dummy_dataset):
    """Verify full end-to-end 5-epoch training loop with plots and cutout figures."""
    trainer = UNetPPTrainer(
        features=(8, 16, 32, 64, 128),
        learning_rate=1e-4,
        device="cpu",
    )
    fixed_val_cutouts = extract_fixed_val_cutouts(dummy_dataset, num_cutouts=4)
    assert len(fixed_val_cutouts) == 4

    def epoch_callback(trainer_inst, epoch_data, history):
        ep = epoch_data["epoch"]
        epochs = epoch_data["epochs"]
        if ep % 5 == 0 or ep == epochs:
            plot_training_curves(
                history=history,
                output_dir=tmp_path,
                current_epoch=ep,
                boundary_tolerance=1,
            )
            fig_path = tmp_path / f"val_inference_epoch_{ep:03d}.png"
            plot_validation_examples(
                model=trainer_inst.model,
                fixed_cutouts=fixed_val_cutouts,
                gpu_transform=trainer_inst.gpu_transform,
                output_path=fig_path,
                epoch=ep,
                epochs=epochs,
                device=trainer_inst.device,
            )

    history = trainer.train(
        dataset=dummy_dataset,
        val_dataset=dummy_dataset,
        epochs=5,
        batch_size=2,
        output_dir=tmp_path,
        log_file="epoch_log.txt",
        epoch_callback=epoch_callback,
        verbose=False,
    )

    # 1. Epoch log file
    log_file = tmp_path / "epoch_log.txt"
    assert log_file.exists()
    lines = [ln for ln in log_file.read_text().splitlines() if "Epoch [" in ln]
    assert len(lines) == 5

    # 2. Training curves at epoch 5
    assert (tmp_path / "gh200_training_curves.png").exists()
    assert (tmp_path / "training_curves_epoch_005.png").exists()

    # 3. Validation inference figure at epoch 5
    assert (tmp_path / "val_inference_epoch_005.png").exists()
    assert (tmp_path / "val_inference_latest.png").exists()

    # 4. JSON metrics
    assert (tmp_path / "training_metrics.json").exists()


def test_train_gh200_cli_expanded_glob():
    """Verify that multiple shards from unquoted shell glob expansion parse successfully."""
    from cloud.train_gh200 import build_parser

    # Simulate unquoted shell glob expansion where shell expands *.tar to multiple arguments
    simulated_args = [
        "--train_shards",
        "train-000001.tar",
        "train-000002.tar",
        "train-000003.tar",
        "--val_shards",
        "val-000001.tar",
        "val-000002.tar",
        "--epochs",
        "1",
        "--batch_size",
        "4",
    ]

    parser = build_parser()
    args = parser.parse_args(simulated_args)

    assert len(args.train_shards) == 3
    assert args.train_shards == ["train-000001.tar", "train-000002.tar", "train-000003.tar"]
    assert len(args.val_shards) == 2
    assert args.val_shards == ["val-000001.tar", "val-000002.tar"]
    assert args.epochs == 1
    assert args.batch_size == 4
    assert args.plot_interval == 5
    assert args.log_file == "epoch_log.txt"


