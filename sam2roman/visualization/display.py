"""
Matplotlib visualization routines for astronomical FITS images and SAM 2 masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from sam2roman.io.fits import FITSImage


def create_distinct_colormap(n_colors: int = 100, seed: int = 42) -> np.ndarray:
    """Generate distinct random RGBA colors for mask visualization."""
    rng = np.random.RandomState(seed)
    colors = rng.rand(n_colors, 3)
    # Ensure colors are bright enough against dark astronomical background
    colors = 0.3 + 0.7 * colors
    alpha = np.full((n_colors, 1), 0.5)
    return np.hstack([colors, alpha])


def plot_masks_overlay(
    image: Union[FITSImage, np.ndarray, Image.Image],
    masks: List[np.ndarray],
    boxes: Optional[List[List[float]]] = None,
    centroids: Optional[List[Tuple[float, float]]] = None,
    stretch_type: str = "leakysb",
    alpha: float = 0.45,
    title: str = "SAM 2 Astronomical Segmentation Overlay",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 10),
    show_labels: bool = True,
) -> plt.Figure:
    """
    Overlay segmentation masks onto the astronomical image.

    Parameters
    ----------
    image : FITSImage, np.ndarray, or PIL Image
        Input astronomical image.
    masks : List[np.ndarray]
        List of 2D boolean masks.
    boxes : Optional[List[List[float]]]
        Optional bounding boxes to draw.
    centroids : Optional[List[Tuple[float, float]]]
        Optional centroids (x, y) to mark.
    stretch_type : str, default 'leakysb'
        Astronomical stretch for 2D arrays ('leakysb', 'asinh', etc.).
    alpha : float, default 0.45
        Transparency of the mask overlay.
    title : str
    save_path : Optional[str or Path]
        Path to save figure.
    figsize : Tuple[int, int]
    show_labels : bool, default True
        Whether to show numerical object ID labels near centroids.
    """
    from sam2roman.io.fits import FITSImage, stretch_astro_image

    if isinstance(image, FITSImage):
        bg = stretch_astro_image(image.data, stretch_type=stretch_type)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            bg = stretch_astro_image(image, stretch_type=stretch_type)
        else:
            bg = image
    elif isinstance(image, Image.Image):
        bg = np.array(image) / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    fig, ax = plt.subplots(figsize=figsize)
    if bg.ndim == 2:
        ax.imshow(bg, origin="lower", cmap="gray")
    else:
        ax.imshow(bg, origin="lower")

    if masks:
        h, w = masks[0].shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        colors = create_distinct_colormap(max(len(masks), 50))

        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            m_bool = mask.astype(bool)
            overlay[m_bool] = color
            overlay[m_bool, 3] = alpha

        ax.imshow(overlay, origin="lower")

    if boxes is not None:
        for b in boxes:
            x0, y0, x1, y1 = b
            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="yellow",
                linewidth=1.2,
                linestyle="--",
            )
            ax.add_patch(rect)

    if centroids is not None:
        for idx, (cx, cy) in enumerate(centroids, start=1):
            ax.plot(cx, cy, "r+", markersize=6, markeredgewidth=1.2)
            if show_labels:
                ax.text(
                    cx + 2,
                    cy + 2,
                    str(idx),
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.6),
                )

    ax.set_title(f"{title} ({len(masks)} objects detected)", fontsize=13)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_detection_summary(
    image: Union[FITSImage, np.ndarray],
    seg_result: Any,
    title_prefix: str = "Roman Field",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Produce a 3-panel astronomical summary:
    1. Raw FITS with astronomical stretch
    2. SAM 2 segmentation overlay
    3. Object area distribution or integer segmentation label map
    """
    from sam2roman.io.fits import FITSImage, stretch_astro_image

    if isinstance(image, FITSImage):
        data = image.data
    else:
        data = image

    stretched = stretch_astro_image(data, stretch_type="leakysb")
    masks = seg_result.masks

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Astronomical Stretch
    axes[0].imshow(stretched, origin="lower", cmap="inferno")
    axes[0].set_title(f"{title_prefix}: Asinh Stretch", fontsize=11)
    axes[0].set_xlabel("Pixel X")
    axes[0].set_ylabel("Pixel Y")

    # Panel 2: SAM 2 Overlaid Masks
    axes[1].imshow(stretched, origin="lower", cmap="gray")
    if masks:
        h, w = masks[0].shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)
        colors = create_distinct_colormap(max(len(masks), 50))
        for idx, m in enumerate(masks):
            color = colors[idx % len(colors)]
            m_bool = m.astype(bool)
            overlay[m_bool] = color
            overlay[m_bool, 3] = 0.5
        axes[1].imshow(overlay, origin="lower")

    if seg_result.centroids:
        for cx, cy in seg_result.centroids:
            axes[1].plot(cx, cy, "r+", markersize=5)

    axes[1].set_title(f"SAM 2 Overlay ({len(masks)} detections)", fontsize=11)
    axes[1].set_xlabel("Pixel X")

    # Panel 3: Labeled Segmentation Map
    comb = seg_result.to_combined_mask()
    im3 = axes[2].imshow(comb, origin="lower", cmap="nipy_spectral")
    axes[2].set_title("Labeled SegMap (Object IDs)", fontsize=11)
    axes[2].set_xlabel("Pixel X")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="Object ID")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_two_panel_prediction(
    stretched: np.ndarray,
    pred_mask: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 150,
    cmap_img: str = "gray",
    image_title: str = "Astronomical Image (LeakySB Stretch)",
    mask_title: str = "Predicted Segmentation Mask (UNet++)",
) -> plt.Figure:
    """
    Display or save a high-contrast side-by-side 2-panel figure showing:
    1. LeakySB-stretched astronomical image.
    2. Boundary-aware 3-class segmentation prediction (Sky, Core, Deblending Edge).

    Parameters
    ----------
    stretched : np.ndarray
        2D normalized [0, 1] astronomical image from leakySB.
    pred_mask : np.ndarray
        2D integer prediction mask with values 0 (Sky), 1 (Core), 2 (Boundary).
    title : str, optional
        Suptitle for the entire figure.
    save_path : str or Path, optional
        File path to save the generated figure.
    figsize : Tuple[int, int], default (16, 8)
        Width and height in inches.
    dpi : int, default 150
        Resolution for saved image.
    cmap_img : str, default 'gray'
        Colormap for the stretched astronomical image.
    image_title : str, default 'Astronomical Image (LeakySB Stretch)'
    mask_title : str, default 'Predicted Segmentation Mask (UNet++)'

    Returns
    -------
    matplotlib.figure.Figure
    """
    h, w = stretched.shape
    total_pixels = h * w
    n_sky = int(np.sum(pred_mask == 0))
    n_core = int(np.sum(pred_mask == 1))
    n_edge = int(np.sum(pred_mask == 2))

    pct_sky = 100.0 * n_sky / total_pixels
    pct_core = 100.0 * n_core / total_pixels
    pct_edge = 100.0 * n_edge / total_pixels

    # Setup discrete 3-class colormap:
    # 0: Sky background (#0b0d1b - Dark charcoal / navy)
    # 1: Core interior (#00d4ff - Electric cyan)
    # 2: Boundary edge (#ff3366 - Vivid coral red)
    cmap_3class = ListedColormap(["#0b0d1b", "#00d4ff", "#ff3366"])
    norm_3class = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_3class.N)

    # Subplots sharing x and y axes for linked interactive zoom/pan
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    # Panel 1: LeakySB stretched image
    im0 = axes[0].imshow(stretched, cmap=cmap_img, origin="lower")
    axes[0].set_title(f"{image_title}\nDimension: {w} × {h} px", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Pixel X", fontsize=10)
    axes[0].set_ylabel("Pixel Y", fontsize=10)
    axes[0].grid(False)

    # Add subtle colorbar for Panel 1
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Normalized SB Flux [0, 1]", fontsize=9)

    # Panel 2: Predicted segmentation mask
    im1 = axes[1].imshow(pred_mask, cmap=cmap_3class, norm=norm_3class, origin="lower")
    axes[1].set_title(
        f"{mask_title}\nCore: {pct_core:.2f}% | Edge: {pct_edge:.2f}% | Sky: {pct_sky:.2f}%",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("Pixel X", fontsize=10)
    axes[1].grid(False)

    # Legend for Panel 2
    patches = [
        mpatches.Patch(color="#0b0d1b", label="Sky Background (0)"),
        mpatches.Patch(color="#00d4ff", label="Source Core (1)"),
        mpatches.Patch(color="#ff3366", label="Deblending Edge (2)"),
    ]
    axes[1].legend(
        handles=patches,
        loc="upper right",
        framealpha=0.85,
        facecolor="#1c2038",
        edgecolor="#444b6e",
        fontsize=9,
        labelcolor="white",
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
