#!/usr/bin/env python3
"""
Interactive WebDataset Cutout Viewer for sam2roman
==================================================

Visualizes cutouts and target masks stored in WebDataset (.tar) shards.

Features:
- Interactive navigation using Left [←] and Right [→] Arrow keys (and GUI buttons)
- 3-Panel scientific layout:
    1. Stretched Astronomical Image (leakySB / physical flux)
    2. Ground Truth Target (Background, Interior Core, Boundary / Edge)
    3. Composite Multi-color Overlay with centroid markers
- Astrometric metadata panel (parent FITS file, crop coordinates, source count, flux statistics)
- Instantaneous back-and-forth browsing with an in-memory sliding cache
- Save view to PNG via [S] key or GUI button
- Headless mode (--save_preview N) for remote SSH / cloud servers without display
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button
import numpy as np
import webdataset as wds

# Ensure sam2roman is imported
for candidate in [Path(__file__).resolve().parent.parent.parent, Path.cwd()]:
    if (candidate / "sam2roman").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sam2roman.io.fits import stretch_astro_image
from sam2roman.training.webdataset_loader import resolve_shard_urls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.view_webdataset")


class WebDatasetViewer:
    """
    Interactive viewer for browsing WebDataset (.tar) shards with Arrow key navigation.
    """

    def __init__(
        self,
        urls: Union[str, Path, Sequence[Union[str, Path]]],
        stretch_type: str = "leakysb",
        zero_points: float = 31.4,
        pixel_scale: float = 0.11,
        mmin: float = 13.0,
        mmax: float = 32.0,
        max_cache: int = 500,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        self.stretch_type = stretch_type
        self.zero_points = zero_points
        self.pixel_scale = pixel_scale
        self.mmin = mmin
        self.mmax = mmax
        self.max_cache = max_cache
        self.save_dir = Path(save_dir or "./previews")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        shard_list = resolve_shard_urls(urls)
        logger.info(f"Connecting to {len(shard_list)} shard archives...")
        self.shard_pipeline = (
            wds.WebDataset(shard_list, shardshuffle=False)
            .decode()
            .to_tuple("raw.npy", "target.npy", "json")
        )
        self.iterator = iter(self.shard_pipeline)
        self.samples: List[Dict[str, Any]] = []
        self.current_idx = 0
        self.has_more = True

        # Pre-fetch first sample
        if not self._fetch_next_sample():
            raise RuntimeError(f"No valid samples found in shard files: {shard_list}")

        # Setup colormap for 3-class segmentation
        # 0: Sky background (dark navy), 1: Core (cyan), 2: Edge (bright coral red)
        self.cmap_3class = ListedColormap(["#0b0d1b", "#00d4ff", "#ff3366"])
        self.norm_3class = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], self.cmap_3class.N)

    def _fetch_next_sample(self) -> bool:
        """Fetch next sample from shard stream and add to sliding cache."""
        if not self.has_more:
            return False
        try:
            raw, target, meta = next(self.iterator)
            sample = {
                "raw": np.asarray(raw, dtype=np.float32),
                "target": np.asarray(target, dtype=np.int64),
                "meta": meta if isinstance(meta, dict) else {},
            }
            self.samples.append(sample)
            if len(self.samples) > self.max_cache:
                self.samples.pop(0)
                self.current_idx = max(0, self.current_idx - 1)
            return True
        except StopIteration:
            self.has_more = False
            return False

    def get_sample(self, index: int) -> Optional[Dict[str, Any]]:
        """Get sample at index, fetching from stream if necessary."""
        while index >= len(self.samples) and self.has_more:
            if not self._fetch_next_sample():
                break
        if 0 <= index < len(self.samples):
            return self.samples[index]
        return None

    def make_overlay(self, raw_stretched: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Create composite RGB image with semi-transparent core & boundary overlay."""
        rgb = np.stack([raw_stretched, raw_stretched, raw_stretched], axis=-1)

        unique_labels = np.unique(target)
        is_3class = set(unique_labels).issubset({0, 1, 2})

        if is_3class:
            is_interior = (target == 1)
            is_boundary = (target == 2)

            # Core tint: cyan (0.0, 0.85, 1.0)
            a_int = 0.45
            rgb[is_interior, 0] = rgb[is_interior, 0] * (1 - a_int) + 0.0 * a_int
            rgb[is_interior, 1] = rgb[is_interior, 1] * (1 - a_int) + 0.85 * a_int
            rgb[is_interior, 2] = rgb[is_interior, 2] * (1 - a_int) + 1.0 * a_int

            # Boundary tint: bright red/coral (1.0, 0.2, 0.3)
            a_bound = 0.85
            rgb[is_boundary, 0] = rgb[is_boundary, 0] * (1 - a_bound) + 1.0 * a_bound
            rgb[is_boundary, 1] = rgb[is_boundary, 1] * (1 - a_bound) + 0.2 * a_bound
            rgb[is_boundary, 2] = rgb[is_boundary, 2] * (1 - a_bound) + 0.3 * a_bound
        else:
            # Multi-instance map: random distinct color per label
            rng = np.random.RandomState(42)
            for lbl in unique_labels:
                if lbl <= 0:
                    continue
                color = rng.uniform(0.3, 1.0, size=3)
                mask = (target == lbl)
                a = 0.5
                for c in range(3):
                    rgb[mask, c] = rgb[mask, c] * (1 - a) + color[c] * a

        return np.clip(rgb, 0.0, 1.0)

    def render_figure(self, fig, axes, cbar_ax=None) -> None:
        """Render the current sample onto figure axes."""
        sample = self.get_sample(self.current_idx)
        if sample is None:
            return

        raw = sample["raw"]
        target = sample["target"]
        meta = sample["meta"]

        for ax in axes:
            ax.clear()

        # 1. Stretched image
        stretched = stretch_astro_image(
            raw,
            stretch_type=self.stretch_type,
            zero_points=self.zero_points,
            pixel_scale=self.pixel_scale,
            mmin=self.mmin,
            mmax=self.mmax,
        )
        axes[0].imshow(stretched, cmap="gray", origin="lower")
        axes[0].set_title(f"Stretched FITS ({self.stretch_type.upper()})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # 2. Target Mask
        unique_labels = np.unique(target)
        is_3class = set(unique_labels).issubset({0, 1, 2})

        if is_3class:
            im_target = axes[1].imshow(target, cmap=self.cmap_3class, norm=self.norm_3class, origin="lower")
            axes[1].set_title("3-Class Target (Sky / Core / Edge)", fontsize=11, fontweight="bold")
        else:
            im_target = axes[1].imshow(target, cmap="nipy_spectral", origin="lower")
            axes[1].set_title(f"Instance Masks ({len(unique_labels)-1} sources)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # 3. Composite Overlay
        overlay = self.make_overlay(stretched, target)
        axes[2].imshow(overlay, origin="lower")
        axes[2].set_title("Detection Overlay", fontsize=11, fontweight="bold")

        # Draw primary centroid marker if available
        if "primary_centroid" in meta:
            cx, cy = meta["primary_centroid"]
            axes[2].plot(cx, cy, "r+", markersize=14, markeredgewidth=2, label="Centroid")
            axes[2].legend(loc="upper right", framealpha=0.6, fontsize=9)
        axes[2].axis("off")

        # Construct informative header
        sample_key = meta.get("sample_key", f"Sample #{self.current_idx + 1}")
        parent_file = meta.get("parent_file", "unknown")
        crop_box = meta.get("crop_box", "N/A")
        num_objs = meta.get("num_objects", len(unique_labels[unique_labels > 0]))
        f_min = meta.get("flux_min", float(np.min(raw)))
        f_max = meta.get("flux_max", float(np.max(raw)))

        header_text = (
            f"[{self.current_idx + 1}/{len(self.samples)}{'+' if self.has_more else ''}]  "
            f"Key: {sample_key}  |  Parent: {parent_file}  |  Crop: {crop_box}\n"
            f"Sources: {num_objs}  |  Flux Range: [{f_min:.1f}, {f_max:.1f}] DN  |  Size: {raw.shape[0]}x{raw.shape[1]}"
        )
        fig.suptitle(header_text, fontsize=11, y=0.98)
        fig.canvas.draw_idle()

    def show(self) -> None:
        """Launch interactive matplotlib viewer with Arrow key listener and GUI buttons."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.15, wspace=0.08)

        # Navigation handlers
        def next_sample(event=None):
            if self.current_idx + 1 < len(self.samples) or self.has_more:
                self.current_idx += 1
                self.render_figure(fig, axes)

        def prev_sample(event=None):
            if self.current_idx > 0:
                self.current_idx -= 1
                self.render_figure(fig, axes)

        def save_current(event=None):
            out_path = self.save_dir / f"cutout_{self.current_idx:05d}.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            logger.info(f"Saved snapshot to: {out_path}")
            print(f"-> Saved current sample view to: {out_path}")

        def random_sample(event=None):
            if len(self.samples) > 1:
                self.current_idx = random.randint(0, len(self.samples) - 1)
                self.render_figure(fig, axes)

        # Keyboard listener
        def on_key(event):
            if event.key in ("right", "down", " ", "enter", "n"):
                next_sample()
            elif event.key in ("left", "up", "backspace", "p"):
                prev_sample()
            elif event.key in ("r",):
                random_sample()
            elif event.key in ("s",):
                save_current()
            elif event.key in ("q", "escape"):
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)

        # GUI Buttons at bottom
        ax_prev = plt.axes([0.18, 0.03, 0.12, 0.06])
        ax_next = plt.axes([0.70, 0.03, 0.12, 0.06])
        ax_rand = plt.axes([0.44, 0.03, 0.12, 0.06])
        ax_save = plt.axes([0.84, 0.03, 0.10, 0.06])

        btn_prev = Button(ax_prev, "◀ Previous [←]", color="#e0e0e0", hovercolor="#c0c0c0")
        btn_next = Button(ax_next, "Next [→] ▶", color="#e0e0e0", hovercolor="#c0c0c0")
        btn_rand = Button(ax_rand, "Random [R]", color="#e0e0e0", hovercolor="#c0c0c0")
        btn_save = Button(ax_save, "Save [S]", color="#d4edda", hovercolor="#c3e6cb")

        btn_prev.on_clicked(prev_sample)
        btn_next.on_clicked(next_sample)
        btn_rand.on_clicked(random_sample)
        btn_save.on_clicked(save_current)

        # Add navigation instructions banner
        fig.text(
            0.50,
            0.11,
            "Navigate with Keyboard [← / →] Arrows  |  [R] Random  |  [S] Save PNG  |  [Q] Quit",
            ha="center",
            fontsize=9.5,
            color="#555555",
            style="italic",
        )

        self.render_figure(fig, axes)
        print("\n" + "=" * 70)
        print("  WebDataset Cutout Viewer Active")
        print("  -> Press [→] Right Arrow for Next Sample")
        print("  -> Press [←] Left Arrow for Previous Sample")
        print("  -> Press [S] to Save PNG Snapshot")
        print("  -> Press [Q] to Exit")
        print("=" * 70 + "\n")
        plt.show()

    def export_previews(self, count: int = 10) -> List[Path]:
        """Save a batch of static PNG previews (useful for headless / cloud runs)."""
        saved = []
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.06, wspace=0.08)

        for i in range(count):
            self.current_idx = i
            sample = self.get_sample(i)
            if sample is None:
                break
            self.render_figure(fig, axes)
            out_file = self.save_dir / f"sample_{i:04d}.png"
            fig.savefig(out_file, dpi=180, bbox_inches="tight")
            saved.append(out_file)
            logger.info(f"Exported preview [{i+1}/{count}]: {out_file.name}")

        plt.close(fig)
        return saved


def main():
    parser = argparse.ArgumentParser(
        description="Interactive WebDataset Cutout Viewer with Arrow Key Navigation."
    )
    parser.add_argument(
        "shards",
        type=str,
        help="Path to .tar shard, shard directory (e.g. data_shards/train), or glob pattern (*.tar).",
    )
    parser.add_argument(
        "--stretch",
        type=str,
        default="leakysb",
        choices=["leakysb", "asinh", "zscale", "linear", "log", "sqrt"],
        help="Astronomical surface brightness stretch (default: 'leakysb').",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./previews",
        help="Directory to save snapshot images (default: ./previews).",
    )
    parser.add_argument(
        "--save_preview",
        type=int,
        default=None,
        metavar="N",
        help="Non-interactive mode: export N sample PNG previews without opening a GUI window.",
    )
    parser.add_argument(
        "--max_cache",
        type=int,
        default=500,
        help="Maximum in-memory sliding window cache size (default: 500).",
    )

    args = parser.parse_args()

    viewer = WebDatasetViewer(
        urls=args.shards,
        stretch_type=args.stretch,
        max_cache=args.max_cache,
        save_dir=args.save_dir,
    )

    if args.save_preview is not None:
        print(f"Exporting {args.save_preview} sample previews to: {args.save_dir}")
        viewer.export_previews(count=args.save_preview)
        print("Done!")
    else:
        viewer.show()


if __name__ == "__main__":
    main()
