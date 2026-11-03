#!/usr/bin/env python3
"""
WebDataset Cutout Materialization Tool for Center Offset Instance Segmentation
=============================================================================

Pre-slices large (e.g. 4088x4088) astronomical FITS images and instance masks into 
high-speed WebDataset (.tar) shards optimized for Grace Hopper (GH200):
- <key>.raw.npy:    Raw float32 detector counts (unclipped physical flux)
- <key>.target.npy: Integer instance segmentation map (0: background, 1..N: source IDs)
- <key>.json:       Metadata with individual object centroids, bounding boxes, and fluxes
- Optional (<key>.center.npy, <key>.offset.npy): Precomputed Gaussian center heatmaps and 2D offset fields

Optimized for sequential zero-seek streaming over NVLink-C2C and high-throughput memory buffers.
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

from sam2roman.training.offset_target import generate_center_offset_targets_numpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam2roman.build_segment_webdataset")


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

    mask_stem_map = {}
    for p in masks_dir.glob(pattern):
        clean_stem = (
            p.stem.replace("_seg", "")
            .replace("_mask", "")
            .replace(".mask", "")
            .replace("mask_", "")
            .replace("_instances", "")
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


def extract_instance_cutouts_from_file(
    image_path: Path,
    mask_path: Path,
    chunk_size: int = 512,
    stride: Optional[int] = None,
    min_objects: int = 1,
    min_pixels_per_object: int = 4,
    precompute_offsets: bool = False,
    center_sigma: float = 2.0,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract fixed-size instance cutouts from a single FITS image/mask pair."""
    if stride is None:
        stride = chunk_size * 3 // 4

    with fits.open(image_path, memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        img_data = np.squeeze(hdu.data).astype(np.float32)

    with fits.open(mask_path, memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        mask_data = np.squeeze(hdu.data).astype(np.int32)

    h, w = img_data.shape
    samples = []

    # Pad if smaller than chunk_size
    if h < chunk_size or w < chunk_size:
        pad_y = max(0, chunk_size - h)
        pad_x = max(0, chunk_size - w)
        img_data = np.pad(img_data, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        mask_data = np.pad(mask_data, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
        h, w = img_data.shape

    for y0 in range(0, h - chunk_size + 1, stride):
        for x0 in range(0, w - chunk_size + 1, stride):
            y1 = y0 + chunk_size
            x1 = x0 + chunk_size

            raw_crop = img_data[y0:y1, x0:x1]
            mask_crop = mask_data[y0:y1, x0:x1].copy()

            unique_labels = np.unique(mask_crop)
            unique_labels = unique_labels[unique_labels > 0]

            # Re-index instance labels to contiguous 1..K inside this cutout
            # while filtering out slivers (< min_pixels_per_object)
            valid_labels = []
            relabeled_crop = np.zeros_like(mask_crop)
            new_id = 1
            centroids = []

            for lbl in unique_labels:
                ys, xs = np.where(mask_crop == lbl)
                if len(ys) >= min_pixels_per_object:
                    relabeled_crop[ys, xs] = new_id
                    valid_labels.append(new_id)
                    cy = float(np.mean(ys))
                    cx = float(np.mean(xs))
                    centroids.append({
                        "id": new_id,
                        "cy": round(cy, 2),
                        "cx": round(cx, 2),
                        "area": int(len(ys)),
                        "bbox": [int(np.min(ys)), int(np.min(xs)), int(np.max(ys)), int(np.max(xs))],
                    })
                    new_id += 1

            if len(valid_labels) < min_objects:
                continue

            meta = {
                "parent_file": image_path.name,
                "crop_box": [int(y0), int(y1), int(x0), int(x1)],
                "chunk_size": int(chunk_size),
                "num_objects": int(len(valid_labels)),
                "centroids": centroids,
                "flux_min": float(np.min(raw_crop)),
                "flux_max": float(np.max(raw_crop)),
                "flux_median": float(np.median(raw_crop)),
            }

            sample_dict: Dict[str, Any] = {
                "raw": raw_crop,
                "target": relabeled_crop.astype(np.int32),
                "meta": meta,
            }

            if precompute_offsets:
                fg, c_map, o_map, _ = generate_center_offset_targets_numpy(
                    relabeled_crop,
                    sigma=center_sigma,
                    min_area=min_pixels_per_object,
                )
                sample_dict["fg"] = fg
                sample_dict["center"] = c_map
                sample_dict["offset"] = o_map

            samples.append(sample_dict)

            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def write_segment_shards(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split_name: str,
    chunk_size: int = 512,
    stride: Optional[int] = None,
    min_objects: int = 1,
    min_pixels_per_object: int = 4,
    precompute_offsets: bool = False,
    center_sigma: float = 2.0,
    max_samples_per_file: Optional[int] = None,
    max_samples_per_shard: int = 1000,
) -> int:
    """Write extracted cutouts into WebDataset .tar shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = str(output_dir / f"{split_name}-%06d.tar")
    total_written = 0

    with wds.ShardWriter(shard_pattern, maxcount=max_samples_per_shard) as sink:
        for idx, (img_p, mask_p) in enumerate(pairs):
            logger.info(
                f"[{split_name.upper()}] Slicing ({chunk_size}x{chunk_size}) exposure "
                f"{idx + 1}/{len(pairs)}: {img_p.name}"
            )
            try:
                samples = extract_instance_cutouts_from_file(
                    image_path=img_p,
                    mask_path=mask_p,
                    chunk_size=chunk_size,
                    stride=stride,
                    min_objects=min_objects,
                    min_pixels_per_object=min_pixels_per_object,
                    precompute_offsets=precompute_offsets,
                    center_sigma=center_sigma,
                    max_samples=max_samples_per_file,
                )
            except Exception as e:
                logger.warning(f"Error processing {img_p.name}: {e}. Skipping exposure.")
                continue

            for s in samples:
                key = f"{split_name}_{total_written:08d}"
                wds_record: Dict[str, Any] = {
                    "__key__": key,
                    "raw.npy": to_npy_bytes(s["raw"]),
                    "target.npy": to_npy_bytes(s["target"]),
                    "json": to_json_bytes(s["meta"]),
                }
                if precompute_offsets:
                    wds_record["fg.npy"] = to_npy_bytes(s["fg"])
                    wds_record["center.npy"] = to_npy_bytes(s["center"])
                    wds_record["offset.npy"] = to_npy_bytes(s["offset"])

                sink.write(wds_record)
                total_written += 1

    logger.info(f"[{split_name.upper()}] Wrote {total_written} cutouts into shards at '{output_dir}'.")
    return total_written


def build_segment_webdataset(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    output_dir: Union[str, Path],
    chunk_size: int = 512,
    stride: Optional[int] = None,
    min_objects: int = 1,
    min_pixels_per_object: int = 4,
    val_ratio: float = 0.2,
    max_images: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
    max_samples_per_shard: int = 1000,
    precompute_offsets: bool = False,
    center_sigma: float = 2.0,
    pattern: str = "*.fits",
    seed: int = 42,
) -> Dict[str, int]:
    """Execute train/val split and materialize shards for center offset segmentation."""
    images_p = Path(images_dir)
    masks_p = Path(masks_dir)
    out_p = Path(output_dir)

    pairs = match_fits_pairs(images_p, masks_p, pattern=pattern)
    if not pairs:
        raise FileNotFoundError(f"No matching FITS image-mask pairs found in '{images_p}' and '{masks_p}'")

    logger.info(f"Matched {len(pairs)} exposure pairs.")
    if max_images is not None and max_images > 0:
        pairs = pairs[:max_images]
        logger.info(f"Capped at {len(pairs)} pairs via max_images.")

    # File-level split to prevent cross-validation data leakage
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = int(round(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Fallback if only 1 file is present
    if len(train_pairs) == 0 and len(val_pairs) > 0:
        train_pairs = val_pairs
        val_pairs = []

    logger.info(f"Split: {len(train_pairs)} training exposures, {len(val_pairs)} validation exposures.")

    train_out = out_p / "train"
    val_out = out_p / "val"

    train_count = write_segment_shards(
        train_pairs,
        output_dir=train_out,
        split_name="train",
        chunk_size=chunk_size,
        stride=stride,
        min_objects=min_objects,
        min_pixels_per_object=min_pixels_per_object,
        precompute_offsets=precompute_offsets,
        center_sigma=center_sigma,
        max_samples_per_file=max_samples_per_file,
        max_samples_per_shard=max_samples_per_shard,
    )

    val_count = 0
    if len(val_pairs) > 0:
        val_count = write_segment_shards(
            val_pairs,
            output_dir=val_out,
            split_name="val",
            chunk_size=chunk_size,
            stride=stride,
            min_objects=min_objects,
            min_pixels_per_object=min_pixels_per_object,
            precompute_offsets=precompute_offsets,
            center_sigma=center_sigma,
            max_samples_per_file=max_samples_per_file,
            max_samples_per_shard=max_samples_per_shard,
        )

    manifest = {
        "images_dir": str(images_p),
        "masks_dir": str(masks_p),
        "total_exposures": len(pairs),
        "train_exposures": len(train_pairs),
        "val_exposures": len(val_pairs),
        "train_cutouts": train_count,
        "val_cutouts": val_count,
        "chunk_size": chunk_size,
        "stride": stride,
        "precompute_offsets": precompute_offsets,
        "center_sigma": center_sigma,
    }
    with open(out_p / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset materialization complete. Total cutouts: {train_count + val_count}.")
    return {"train_cutouts": train_count, "val_cutouts": val_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize astronomical instance segmentation cutouts into high-speed WebDataset shards.",
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing raw FITS exposures.")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory containing FITS instance masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination directory for WebDataset shards.")
    parser.add_argument("--chunk_size", type=int, default=512, help="Cutout dimension H=W in pixels (default: 512).")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride (default: 3/4 chunk_size).")
    parser.add_argument("--min_objects", type=int, default=1, help="Minimum detected sources in cutout (default: 1).")
    parser.add_argument("--min_pixels_per_object", type=int, default=4, help="Min pixels per object (default: 4).")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation exposure split fraction (default: 0.2).")
    parser.add_argument("--max_images", type=int, default=None, help="Cap on input exposures.")
    parser.add_argument("--samples_per_file", type=int, default=None, help="Max cutouts per exposure file.")
    parser.add_argument("--max_samples_per_shard", type=int, default=1000, help="Cutouts per .tar shard (default: 1000).")
    parser.add_argument("--precompute_offsets", action="store_true", default=False, help="Precompute center heatmaps and offsets inside shards.")
    parser.add_argument("--center_sigma", type=float, default=2.0, help="Gaussian sigma for center heatmaps (default: 2.0).")
    parser.add_argument("--pattern", type=str, default="*.fits", help="File glob pattern (default: '*.fits').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    build_segment_webdataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        min_objects=args.min_objects,
        min_pixels_per_object=args.min_pixels_per_object,
        val_ratio=args.val_ratio,
        max_images=args.max_images,
        max_samples_per_file=args.samples_per_file,
        max_samples_per_shard=args.max_samples_per_shard,
        precompute_offsets=args.precompute_offsets,
        center_sigma=args.center_sigma,
        pattern=args.pattern,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
