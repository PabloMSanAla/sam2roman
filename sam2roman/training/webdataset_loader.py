"""
High-Performance WebDataset Streaming Loader for sam2roman
===========================================================

Provides sequential, zero-seek streaming DataLoader for WebDataset (.tar) shards:
- In-memory shuffle buffers for i.i.d. sampling
- Transparent multi-worker process splitting
- Native PyTorch tensor conversion and batch collation
- Optimized for direct DMA transfers over NVLink-C2C on NVIDIA Grace Hopper (GH200)
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
import webdataset as wds

logger = logging.getLogger(__name__)


def collate_webdataset_batch(batch: List[Sequence[Any]]) -> Dict[str, Any]:
    """
    Collate function for WebDataset samples tuple: (raw_array, target_array, meta_dict).

    Returns
    -------
    dict:
        "raw": torch.FloatTensor of shape (B, 1, H, W)
        "target": torch.LongTensor of shape (B, H, W)
        "meta": list of sample metadata dicts
    """
    raw_list = []
    target_list = []
    meta_list = []

    for item in batch:
        raw_arr = item[0]
        tgt_arr = item[1]
        meta = item[2] if len(item) > 2 else {}

        # Ensure 2D (H, W)
        if raw_arr.ndim == 2:
            t_raw = torch.from_numpy(raw_arr).unsqueeze(0).float()
        elif raw_arr.ndim == 3 and raw_arr.shape[0] == 1:
            t_raw = torch.from_numpy(raw_arr).float()
        else:
            t_raw = torch.from_numpy(raw_arr).float()
            if t_raw.ndim == 2:
                t_raw = t_raw.unsqueeze(0)

        # Support continuous soft targets (e.g. float32 probability maps or distance fields)
        # as well as discrete integer class masks
        if np.issubdtype(tgt_arr.dtype, np.floating) or (tgt_arr.ndim == 3 and tgt_arr.shape[0] > 1):
            t_tgt = torch.from_numpy(tgt_arr).float()
        else:
            t_tgt = torch.from_numpy(tgt_arr).long()
            if t_tgt.ndim == 3 and t_tgt.shape[0] == 1:
                t_tgt = t_tgt.squeeze(0)

        raw_list.append(t_raw)
        target_list.append(t_tgt)
        meta_list.append(meta)

    return {
        "raw": torch.stack(raw_list, dim=0),      # (B, 1, H, W)
        "target": torch.stack(target_list, dim=0), # (B, H, W)
        "meta": meta_list,
    }


def resolve_shard_urls(urls: Union[str, Path, Sequence[Union[str, Path]]]) -> List[str]:
    """
    Expand directory paths, glob patterns, or URL lists into valid WebDataset shard strings.
    """
    if isinstance(urls, (str, Path)):
        p = Path(urls)
        if p.is_dir():
            # Directory provided: search for all .tar files
            found = sorted([str(f) for f in p.glob("*.tar")])
            if not found:
                # Also check immediate subdirectories (e.g. train/ or val/)
                found = sorted([str(f) for f in p.glob("*/*.tar")])
            if not found:
                raise FileNotFoundError(f"No .tar shard files found inside directory '{urls}'")
            return found
        else:
            # String with glob or single file
            url_str = str(urls)
            if "*" in url_str:
                found = sorted(glob.glob(url_str))
                if not found:
                    raise FileNotFoundError(f"No shard files matched pattern '{url_str}'")
                return found
            else:
                if not Path(url_str).exists():
                    raise FileNotFoundError(f"Shard file not found: '{url_str}'")
                return [url_str]
    else:
        # Sequence of paths/strings
        resolved = []
        for u in urls:
            resolved.extend(resolve_shard_urls(u))
        return resolved


def create_webdataset_loader(
    urls: Union[str, Path, Sequence[Union[str, Path]]],
    batch_size: int = 16,
    is_train: bool = True,
    shuffle_buffer: int = 1000,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    drop_last: bool = True,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
    """
    Build a high-performance streaming DataLoader over WebDataset shards.

    Parameters
    ----------
    urls : str, Path, or list
        Shard path pattern (e.g., 'data/shards/train-*.tar' or directory path).
    batch_size : int, default 16
        Number of cutouts per mini-batch. Set to 32 or 64 on Grace Hopper H100.
    is_train : bool, default True
        If True, enables shard-level shuffling and in-memory sample shuffling.
    shuffle_buffer : int, default 1000
        Size of the in-memory window for i.i.d. random sample extraction.
    num_workers : int, default 4
        Background subprocesses for prefetching. On GH200 (72 ARM cores), 8–16
        workers is the sweet spot. Note: num_workers should be <= number of shards.
    pin_memory : bool, optional
        Pin memory in host RAM for fast non-blocking DMA copies to GPU. Defaults to True if CUDA is available.
    drop_last : bool, default True
        Drop incomplete final batch.
    prefetch_factor : int, optional
        Number of batches loaded in advance by each worker (default: 2 when num_workers > 0).
    persistent_workers : bool, optional
        Maintain worker processes across epochs to avoid re-spawn latency (default: True when num_workers > 0).

    Returns
    -------
    DataLoader:
        PyTorch DataLoader streaming {'raw': (B, 1, H, W), 'target': (B, H, W), 'meta': list}
    """
    shard_list = resolve_shard_urls(urls)
    logger.info(f"Initialized WebDataset loader with {len(shard_list)} shards (is_train={is_train}).")

    # Shard-to-worker check
    if num_workers > len(shard_list):
        logger.warning(
            f"[sam2roman] num_workers ({num_workers}) exceeds number of available shards ({len(shard_list)}). "
            f"WebDataset distributes whole shards across workers, so {num_workers - len(shard_list)} workers will "
            f"receive 0 shards and remain idle! To fully utilize {num_workers} workers, re-build WebDataset with a "
            f"smaller --samples_per_file in sam2roman/tools/build_webdataset.py to generate >= {num_workers} shards."
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Construct WebDataset streaming pipeline
    shard_shuffle_val = 100 if is_train else False
    pipeline = wds.WebDataset(shard_list, shardshuffle=shard_shuffle_val)

    if is_train and shuffle_buffer > 0:
        pipeline = pipeline.shuffle(shuffle_buffer)

    # Decode .npy and .json directly into NumPy arrays and dicts
    pipeline = pipeline.decode().to_tuple("raw.npy", "target.npy", "json")

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collate_webdataset_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last if is_train else False,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True if persistent_workers is None else persistent_workers
        loader_kwargs["prefetch_factor"] = 2 if prefetch_factor is None else prefetch_factor

    loader = DataLoader(pipeline, **loader_kwargs)

    return loader
