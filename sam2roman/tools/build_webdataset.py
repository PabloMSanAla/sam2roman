#!/usr/bin/env python3
"""
WebDataset Cutout Materialization Tool for sam2roman
====================================================

Pre-slices large (4088x4088) astronomical FITS images and masks into 
high-speed WebDataset (.tar) shards containing:
- <key>.raw.npy:    Raw float32 detector counts (unclipped physical flux)
- <key>.target.npy: 3-class boundary target (0: sky, 1: core, 2: edge) or instance masks
- <key>.json:       Astrometric metadata, crop box, and source statistics

Supports:
- Configurable chunk/cutout size (--chunk_size 512 or 1024)
- Multi-core parallel exposure processing
- Leak-free train/validation file-level splitting
- Zero random seek sequential streaming for NVIDIA Grace Hopper (GH200) and GPU clusters
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import webdataset as wds
from astropy.io import fits

from sam2roman.training.boundary import generate_3class_boundary_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.build_webdataset")


def to_npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a NumPy array to binary bytes in .npy format."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def to_json_bytes(meta: Dict[str, Any]) -> bytes:
    """Serialize a dictionary to UTF-8 JSON bytes."""
    return json.dumps(meta, indent=2).encode("utf-8")


def match_fits_pairs(
    images_dir: Path,
    masks_dir: Path,
    pattern: str = "*.fits",
) -> List[Tuple[Path, Path]]:
    """Match original FITS images with their corresponding mask files."""
    img_files = sorted(list(images_dir.glob(pattern)))
    mask_files = {p.name: p for p in masks_dir.glob(pattern)}

    # Also build dictionary stripping common prefixes/suffixes
    mask_stem_map = {}
    for p in masks_dir.glob(pattern):
        clean_stem = (
            p.stem.replace("_seg", "")
            .replace("_mask", "")
            .replace(".mask", "")
            .replace("mask_", "")
        )
        mask_stem_map[clean_stem] = p

    pairs = []
    for img_p in img_files:
        if img_p.name in mask_files:
            pairs.append((img_p, mask_files[img_p.name]))
            continue

        clean_img_stem = (
            img_p.stem.replace("_orig", "")
            .replace("_original", "")
            .replace(".original", "")
            .replace("original_", "")
        )
        if clean_img_stem in mask_stem_map:
            pairs.append((img_p, mask_stem_map[clean_img_stem]))
            continue

        for m_name, m_path in mask_files.items():
            if img_p.stem in m_name or clean_img_stem in m_name:
                pairs.append((img_p, m_path))
                break

    return pairs


def extract_cutouts_from_file(
    image_path: Path,
    mask_path: Path,
    chunk_size: int = 512,
    stride: Optional[int] = None,
    min_objects: int = 1,
    target_mode: str = "dense_3class",
    boundary_width: int = 2,
    soft_boundary: bool = False,
    boundary_sigma: float = 1.2,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract fixed-size cutouts from a single FITS image/mask pair.
    Returns a list of sample dictionaries ready for serialization.
    """
    if stride is None:
        stride = chunk_size * 3 // 4

    # Load 2D arrays from FITS
    with fits.open(image_path, memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        img_data = np.squeeze(hdu.data).astype(np.float32)

    with fits.open(mask_path, memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        mask_data = np.squeeze(hdu.data).astype(np.int32)

    h, w = img_data.shape
    samples = []

    # If image is smaller than chunk size, pad
    if h < chunk_size or w < chunk_size:
        pad_y = max(0, chunk_size - h)
        pad_x = max(0, chunk_size - w)
        img_data = np.pad(img_data, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        mask_data = np.pad(mask_data, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        h, w = img_data.shape

    # Grid slicing
    for y0 in range(0, h - chunk_size + 1, stride):
        for x0 in range(0, w - chunk_size + 1, stride):
            y1 = y0 + chunk_size
            x1 = x0 + chunk_size

            raw_crop = img_data[y0:y1, x0:x1]
            mask_crop = mask_data[y0:y1, x0:x1]

            unique_labels = np.unique(mask_crop)
            unique_labels = unique_labels[unique_labels > 0]

            if len(unique_labels) < min_objects:
                continue

            if target_mode == "dense_3class":
                target_crop = generate_3class_boundary_mask(
                    mask_crop,
                    boundary_width=boundary_width,
                    soften=soft_boundary,
                    sigma=boundary_sigma,
                )
                if not soft_boundary:
                    target_crop = target_crop.astype(np.int64)
            else:
                target_crop = mask_crop.astype(np.int64)

            # Metadata
            primary_lbl = int(unique_labels[0])
            ys_obj, xs_obj = np.where(mask_crop == primary_lbl)
            cx = float(np.mean(xs_obj)) if len(xs_obj) > 0 else float(chunk_size // 2)
            cy = float(np.mean(ys_obj)) if len(ys_obj) > 0 else float(chunk_size // 2)

            meta = {
                "parent_file": image_path.name,
                "crop_box": [int(y0), int(y1), int(x0), int(x1)],
                "chunk_size": int(chunk_size),
                "num_objects": int(len(unique_labels)),
                "primary_centroid": [round(cx, 2), round(cy, 2)],
                "flux_min": float(np.min(raw_crop)),
                "flux_max": float(np.max(raw_crop)),
                "flux_median": float(np.median(raw_crop)),
            }

            samples.append({
                "raw": raw_crop,
                "target": target_crop,
                "meta": meta,
            })

            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def write_shard_dataset(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split_name: str,
    chunk_size: int = 512,
    stride: Optional[int] = None,
    min_objects: int = 1,
    target_mode: str = "dense_3class",
    boundary_width: int = 2,
    soft_boundary: bool = False,
    boundary_sigma: float = 1.2,
    max_samples_per_file: Optional[int] = None,
    max_samples_per_shard: int = 1000,
) -> int:
    """Write extracted cutouts from image-mask pairs into WebDataset .tar shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_dir / f"{split_name}-%06d.tar")
    total_written = 0

    with wds.ShardWriter(shard_pattern, maxcount=max_samples_per_shard) as sink:
        for idx, (img_p, mask_p) in enumerate(pairs):
            logger.info(
                f"[{split_name.upper()}] Slicing ({chunk_size}x{chunk_size}) exposure "
                f"{idx + 1}/{len(pairs)}: {img_p.name}"
            )
            samples = extract_cutouts_from_file(
                image_path=img_p,
                mask_path=mask_p,
                chunk_size=chunk_size,
                stride=stride,
                min_objects=min_objects,
                target_mode=target_mode,
                boundary_width=boundary_width,
                soft_boundary=soft_boundary,
                boundary_sigma=boundary_sigma,
                max_samples=max_samples_per_file,
            )

            for s in samples:
                sample_key = f"{split_name}_{total_written:08d}"
                s["meta"]["sample_key"] = sample_key

                sink.write({
                    "__key__": sample_key,
                    "raw.npy": to_npy_bytes(s["raw"]),
                    "target.npy": to_npy_bytes(s["target"]),
                    "json": to_json_bytes(s["meta"]),
                })
                total_written += 1

    logger.info(
        f"[{split_name.upper()}] Successfully packaged {total_written} cutouts "
        f"into shards at: {output_dir}"
    )
    return total_written


def build_webdataset(
    images_dir: Path | str,
    masks_dir: Path | str,
    output_dir: Path | str,
    chunk_size: int = 512,
    stride: Optional[int] = None,
    min_objects: int = 1,
    val_ratio: float = 0.2,
    max_images: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
    max_samples_per_shard: int = 1000,
    target_mode: str = "dense_3class",
    boundary_width: int = 2,
    soft_boundary: bool = False,
    boundary_sigma: float = 1.2,
    pattern: str = "*.fits",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main entry point for building high-speed WebDataset shards from FITS directories.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    pairs = match_fits_pairs(images_dir, masks_dir, pattern=pattern)
    if not pairs:
        raise ValueError(f"No matching FITS pairs found between '{images_dir}' and '{masks_dir}'")

    if max_images is not None and max_images > 0:
        logger.info(f"Limiting dataset to first {max_images} input exposure images (out of {len(pairs)} matched).")
        pairs = pairs[:max_images]

    logger.info(f"Matched {len(pairs)} exposure pairs. Splitting train/val with seed={seed}...")
    random.seed(seed)
    shuffled_pairs = list(pairs)
    random.shuffle(shuffled_pairs)

    n_val = max(1, int(len(shuffled_pairs) * val_ratio)) if (len(shuffled_pairs) > 1 and val_ratio > 0.0) else 0
    n_train = len(shuffled_pairs) - n_val

    train_pairs = shuffled_pairs[:n_train]
    val_pairs = shuffled_pairs[n_train:]

    logger.info(
        f"Dataset split: {len(train_pairs)} train exposures ({100*(1-val_ratio):.0f}%), "
        f"{len(val_pairs)} validation exposures ({100*val_ratio:.0f}%)"
    )

    # Build train shards
    train_dir = output_dir / "train"
    train_count = write_shard_dataset(
        pairs=train_pairs,
        output_dir=train_dir,
        split_name="train",
        chunk_size=chunk_size,
        stride=stride,
        min_objects=min_objects,
        target_mode=target_mode,
        boundary_width=boundary_width,
        soft_boundary=soft_boundary,
        boundary_sigma=boundary_sigma,
        max_samples_per_file=max_samples_per_file,
        max_samples_per_shard=max_samples_per_shard,
    )

    # Build val shards
    val_count = 0
    if val_pairs:
        val_dir = output_dir / "val"
        val_count = write_shard_dataset(
            pairs=val_pairs,
            output_dir=val_dir,
            split_name="val",
            chunk_size=chunk_size,
            stride=stride,
            min_objects=min_objects,
            target_mode=target_mode,
            boundary_width=boundary_width,
            soft_boundary=soft_boundary,
            boundary_sigma=boundary_sigma,
            max_samples_per_file=max_samples_per_file,
            max_samples_per_shard=max_samples_per_shard,
        )

    summary = {
        "output_dir": str(output_dir),
        "chunk_size": chunk_size,
        "stride": stride or (chunk_size * 3 // 4),
        "target_mode": target_mode,
        "max_images": max_images,
        "train_exposures": len(train_pairs),
        "val_exposures": len(val_pairs),
        "train_cutouts": train_count,
        "val_cutouts": val_count,
        "total_cutouts": train_count + val_count,
    }

    # Save summary manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Manifest written to: {manifest_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Materialize Roman FITS images into WebDataset (.tar) shards for Grace Hopper training."
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing original FITS images.")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory containing mask FITS images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to store WebDataset shards.")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        choices=[256, 512, 1024, 2048],
        help="Dimension of square cutouts (default: 512; choose 1024 for native SAM 2 ViT resolution).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between neighboring cutouts (default: chunk_size * 3 // 4).",
    )
    parser.add_argument(
        "--min_objects",
        type=int,
        default=1,
        help="Minimum number of detected sources required in a cutout (default: 1).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of exposures allocated to validation (default: 0.2).",
    )
    parser.add_argument(
        "--max_images",
        "--max_files",
        type=int,
        default=None,
        dest="max_images",
        help="Maximum number of input exposure images (FITS file pairs) to slice (default: None for all).",
    )
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=None,
        help="Max cutouts per exposure file (default: None for full coverage).",
    )
    parser.add_argument(
        "--max_samples_per_shard",
        type=int,
        default=1000,
        help="Max number of cutouts packed into each .tar shard file (default: 1000).",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="dense_3class",
        choices=["dense_3class", "instance"],
        help="Target type: 'dense_3class' (0: sky, 1: core, 2: edge) or 'instance' (IDs).",
    )
    parser.add_argument(
        "--boundary_width",
        type=int,
        default=2,
        help="Pixel width of source boundary ring (default: 2).",
    )
    parser.add_argument(
        "--soft_boundary",
        action="store_true",
        default=False,
        help="Smooth boundary target with Gaussian filter into continuous probability simplex (3, H, W).",
    )
    parser.add_argument(
        "--boundary_sigma",
        type=float,
        default=1.2,
        help="Gaussian standard deviation for soft boundary smoothing (default: 1.2).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.fits",
        help="Glob pattern for FITS files (default: '*.fits').",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")

    args = parser.parse_args()

    build_webdataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        min_objects=args.min_objects,
        val_ratio=args.val_ratio,
        max_images=args.max_images,
        max_samples_per_file=args.samples_per_file,
        max_samples_per_shard=args.max_samples_per_shard,
        target_mode=args.target_mode,
        boundary_width=args.boundary_width,
        soft_boundary=args.soft_boundary,
        boundary_sigma=args.boundary_sigma,
        pattern=args.pattern,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
