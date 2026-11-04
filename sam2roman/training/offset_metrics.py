"""
Evaluation metrics for Center Offset Voting instance segmentation.
==================================================================

Includes:
1. Center detection Precision, Recall, F1, and mean localization error (MAE).
2. Offset Endpoint Error (EPE) on foreground source pixels.
3. Fast instance matching and Panoptic Quality (PQ = SQ * RQ).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndimage
import torch


def extract_center_peaks(
    center_prob: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.3,
    min_distance: int = 3,
) -> List[Tuple[float, float, float]]:
    """
    Extract 2D local maxima peaks from center probability heatmap via max-pooling NMS.

    Parameters
    ----------
    center_prob : np.ndarray or torch.Tensor
        Center probability array (H, W) in [0.0, 1.0].
    threshold : float, default 0.3
        Probability cutoff to consider a candidate peak.
    min_distance : int, default 3
        Footprint radius for local non-maximum suppression.

    Returns
    -------
    List[Tuple[float, float, float]]:
        List of (cy, cx, score) for each detected peak.
    """
    if isinstance(center_prob, torch.Tensor):
        c_np = center_prob.detach().float().cpu().numpy()
    else:
        c_np = np.asarray(center_prob, dtype=np.float32)

    if c_np.ndim == 3 and c_np.shape[0] == 1:
        c_np = c_np[0]

    ksize = 2 * min_distance + 1
    max_filtered = ndimage.maximum_filter(c_np, size=ksize, mode="constant")
    is_peak = (c_np == max_filtered) & (c_np >= threshold)

    ys, xs = np.where(is_peak)
    scores = c_np[ys, xs]

    # Sort descending by peak score
    order = np.argsort(-scores)
    peaks = [(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]
    return peaks


def evaluate_center_detection(
    pred_peaks: List[Tuple[float, float, float]],
    gt_centroids: List[Tuple[float, float]],
    tolerance: float = 2.0,
) -> Dict[str, float]:
    """
    Evaluate center detection performance via bipartite greedy distance matching.

    Parameters
    ----------
    pred_peaks : List[Tuple[float, float, float]]
        Predicted peaks (cy, cx, score).
    gt_centroids : List[Tuple[float, float]]
        Ground truth source centroids (cy, cx).
    tolerance : float, default 2.0
        Maximum Euclidean distance in pixels to count as True Positive.

    Returns
    -------
    Dict[str, float]:
        'precision', 'recall', 'f1', 'mae', 'tp', 'fp', 'fn'
    """
    if len(gt_centroids) == 0:
        fp = len(pred_peaks)
        return {"precision": 0.0 if fp > 0 else 1.0, "recall": 1.0, "f1": 0.0, "mae": 0.0, "tp": 0, "fp": fp, "fn": 0}
    if len(pred_peaks) == 0:
        fn = len(gt_centroids)
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "mae": 0.0, "tp": 0, "fp": 0, "fn": fn}

    gt_matched = [False] * len(gt_centroids)
    tp = 0
    fp = 0
    errors: List[float] = []

    for py, px, _ in pred_peaks:
        best_dist = float("inf")
        best_gt_idx = -1
        for gt_idx, (gy, gx) in enumerate(gt_centroids):
            if gt_matched[gt_idx]:
                continue
            dist = np.sqrt((py - gy) ** 2 + (px - gx) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_dist <= tolerance:
            gt_matched[best_gt_idx] = True
            tp += 1
            errors.append(float(best_dist))
        else:
            fp += 1

    fn = len(gt_centroids) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    mae = float(np.mean(errors)) if errors else 0.0

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mae": float(mae),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_offset_epe(
    pred_offset: Union[np.ndarray, torch.Tensor],
    target_offset: Union[np.ndarray, torch.Tensor],
    fg_mask: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute Endpoint Error (EPE) on foreground source pixels.
    EPE = mean(sqrt((dy - dy_gt)^2 + (dx - dx_gt)^2)).

    Supports both single cutouts of shape (2, H, W) and batches of shape (B, 2, H, W).
    """
    if isinstance(pred_offset, torch.Tensor):
        po = pred_offset.detach().float().cpu().numpy()
        to = target_offset.detach().float().cpu().numpy()
        fg = fg_mask.detach().float().cpu().numpy()
    else:
        po = np.asarray(pred_offset, dtype=np.float32)
        to = np.asarray(target_offset, dtype=np.float32)
        fg = np.asarray(fg_mask, dtype=np.float32)

    if po.ndim == 4:
        # Batch: (B, 2, H, W) -> Channel 0: dy, Channel 1: dx
        dy_err = po[:, 0] - to[:, 0]
        dx_err = po[:, 1] - to[:, 1]
        if fg.ndim == 4 and fg.shape[1] == 1:
            mask = fg[:, 0] > 0
        else:
            mask = fg > 0
    elif po.ndim == 3:
        # Single sample: (2, H, W)
        dy_err = po[0] - to[0]
        dx_err = po[1] - to[1]
        if fg.ndim == 3 and fg.shape[0] == 1:
            mask = fg[0] > 0
        else:
            mask = fg > 0
    else:
        raise ValueError(f"Expected 3D (2, H, W) or 4D (B, 2, H, W) offsets, got shape {po.shape}")

    epe_map = np.sqrt(dy_err ** 2 + dx_err ** 2)

    if np.any(mask):
        return float(np.mean(epe_map[mask]))
    return 0.0
