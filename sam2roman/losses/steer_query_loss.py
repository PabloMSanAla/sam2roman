"""
Loss functions and optimizer configuration for SteerQuery Instance Segmentation.
================================================================================

Includes:
1. SteerQueryLoss:
   - Multi-class Cross-Entropy + Dice for 3-class semantics (sky/galaxy, interior, edge).
   - CenterNet penalty-reduced focal loss for star-center heatmap.
   - Weighted BCE for spike-bridging gate G_fg with spike-over-galaxy upweighting.
   - Weighted BCE + Dice for matched star instances.
2. build_steer_query_optimizer:
   - Two-speed AdamW optimizer with 0.1x learning rate and 0 weight decay
     on raw_dtheta and raw_perp_shifts geometric steering parameters.
3. Target generation helper render_star_center_heatmap.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def render_star_center_heatmap(
    inst_gt: torch.Tensor,
    sigma: float = 2.0,
) -> Tuple[torch.Tensor, List[List[Tuple[float, float]]]]:
    """
    Render 2D Gaussian star-center heatmaps at geometric centroids of star instances.

    Parameters
    ----------
    inst_gt : torch.Tensor
        Integer instance mask of shape (B, H, W) where 0 is background/galaxy and k >= 1 are stars.
    sigma : float, default 2.0
        Gaussian standard deviation in pixels.

    Returns
    -------
    Tuple[torch.Tensor, List[List[Tuple[float, float]]]]:
        - H_star_gt: Gaussian heatmap of shape (B, 1, H, W) in [0.0, 1.0].
        - batch_centroids: List of lists containing (cx, cy) centroids for each star in each batch sample.
    """
    b, h, w = inst_gt.shape
    device = inst_gt.device
    dtype = torch.float32

    H_star_gt = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
    batch_centroids: List[List[Tuple[float, float]]] = []

    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    radius = int(math.ceil(3.0 * sigma))

    for i in range(b):
        mask_i = inst_gt[i]
        unique_ids = torch.unique(mask_i)
        unique_ids = unique_ids[unique_ids > 0]
        centroids_i: List[Tuple[float, float]] = []

        for inst_id in unique_ids:
            ys, xs = torch.where(mask_i == inst_id)
            if len(ys) == 0:
                continue
            cy = float(ys.float().mean().item())
            cx = float(xs.float().mean().item())
            centroids_i.append((cx, cy))

            # Render Gaussian patch within 3*sigma
            y_min = max(0, int(math.floor(cy - radius)))
            y_max = min(h, int(math.ceil(cy + radius + 1)))
            x_min = max(0, int(math.floor(cx - radius)))
            x_max = min(w, int(math.ceil(cx + radius + 1)))

            sub_y = y_grid[y_min:y_max, x_min:x_max]
            sub_x = x_grid[y_min:y_max, x_min:x_max]
            dist_sq = (sub_y - cy) ** 2 + (sub_x - cx) ** 2
            patch = torch.exp(-0.5 * dist_sq / (sigma ** 2))

            H_star_gt[i, 0, y_min:y_max, x_min:x_max] = torch.maximum(
                H_star_gt[i, 0, y_min:y_max, x_min:x_max], patch
            )

            # Ensure exact center pixel is 1.0
            cy_int = int(min(h - 1, max(0, round(cy))))
            cx_int = int(min(w - 1, max(0, round(cx))))
            H_star_gt[i, 0, cy_int, cx_int] = 1.0

        batch_centroids.append(centroids_i)

    return H_star_gt, batch_centroids


def compute_spike_over_galaxy_weights(
    inst_gt: torch.Tensor,
    sem_gt: torch.Tensor,
    image: Optional[torch.Tensor] = None,
    upweight_factor: float = 3.0,
) -> torch.Tensor:
    """
    Compute pixel loss weight map w_pix:
    - Base weight = 1.0
    - Upweight = upweight_factor (default 3.0) on high-flux background (unmasked galaxies)
      and on diffraction spikes crossing those high-flux regions.

    Parameters
    ----------
    inst_gt : torch.Tensor
        Instance ground truth of shape (B, H, W).
    sem_gt : torch.Tensor
        Semantic ground truth of shape (B, H, W) (0: sky/galaxy, 1: interior, 2: edge/spike).
    image : torch.Tensor, optional
        Input image flux of shape (B, C, H, W) or (B, 1, H, W).
    upweight_factor : float, default 3.0
    """
    b, h, w = inst_gt.shape
    device = inst_gt.device
    dtype = torch.float32

    w_pix = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    if image is None:
        return w_pix

    # Mean intensity across channels
    if image.ndim == 4:
        flux = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    else:
        flux = image.unsqueeze(1)

    for i in range(b):
        f_i = flux[i, 0]
        med = torch.median(f_i)
        std = torch.std(f_i)
        high_flux = f_i > (med + 2.0 * std)

        inst_i = inst_gt[i]
        sem_i = sem_gt[i]

        # 1. Unmasked high-flux background galaxies (inst_gt == 0)
        unmasked_galaxies = (inst_i == 0) & high_flux
        # 2. Spike pixels (inst_gt > 0 and sem_gt == 2) overlapping high-flux regions
        spike_pixels = (inst_i > 0) & (sem_i == 2) & high_flux

        boost_mask = unmasked_galaxies | spike_pixels
        w_pix[i, 0][boost_mask] = upweight_factor

    return w_pix


class SteerQueryLoss(nn.Module):
    """
    Multi-task loss for SteerQuery:
    L_total = L_sem + L_ctr + L_gate + lambda_inst * (L_inst_bce + L_inst_dice)
    """

    def __init__(
        self,
        sem_weight: float = 1.0,
        center_weight: float = 2.0,
        gate_weight: float = 1.0,
        inst_weight: float = 1.0,
        center_alpha: float = 2.0,
        center_beta: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sem_weight = sem_weight
        self.center_weight = center_weight
        self.gate_weight = gate_weight
        self.inst_weight = inst_weight
        self.center_alpha = center_alpha
        self.center_beta = center_beta
        self.eps = eps

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        predictions: Dict[str, Any],
        inst_gt: torch.Tensor,
        sem_gt: torch.Tensor,
        H_star_gt: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss components.

        Parameters
        ----------
        predictions : Dict[str, Any]
            Outputs from UNetPlusPlusSteerQuery.
        inst_gt : torch.Tensor
            Integer star instance IDs (B, H, W).
        sem_gt : torch.Tensor
            3-class semantic segmentation target (B, H, W).
        H_star_gt : torch.Tensor, optional
            Precomputed Gaussian center heatmap (B, 1, H, W).
        image : torch.Tensor, optional
            Input image for high-flux galaxy weighting.
        """
        logits_sem = predictions["logits_sem"]
        H_star = predictions["H_star"]
        M_inst = predictions["M_inst"]  # (B, N, H, W)
        G_fg = predictions["G_fg"]      # (B, 1, H, W)
        star_coords = predictions["star_coords"]  # (B, N, 2) [x, y]

        b, _, h, w = logits_sem.shape
        device = logits_sem.device

        # If H_star_gt not provided, render dynamically
        if H_star_gt is None:
            H_star_gt, batch_centroids = render_star_center_heatmap(inst_gt, sigma=2.0)
        else:
            batch_centroids = []
            for i in range(b):
                unique_ids = torch.unique(inst_gt[i])
                unique_ids = unique_ids[unique_ids > 0]
                pts = []
                for uid in unique_ids:
                    ys, xs = torch.where(inst_gt[i] == uid)
                    if len(ys) > 0:
                        pts.append((float(xs.float().mean().item()), float(ys.float().mean().item())))
                batch_centroids.append(pts)

        # 1. Semantic Loss: 3-class Cross-Entropy + Dice
        sem_target = sem_gt.long()
        ce_loss = self.ce_loss(logits_sem, sem_target).mean()

        # Multiclass Dice Loss
        prob_sem = F.softmax(logits_sem, dim=1)
        sem_one_hot = F.one_hot(sem_target, num_classes=3).permute(0, 3, 1, 2).float()
        intersection = (prob_sem * sem_one_hot).sum(dim=(2, 3))
        cardinality = prob_sem.sum(dim=(2, 3)) + sem_one_hot.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        l_sem = ce_loss + dice_loss.mean()

        # 2. Center Heatmap Focal Loss (Zhou et al., CenterNet)
        H_pred = H_star.clamp(self.eps, 1.0 - self.eps)
        H_tgt = H_star_gt.clamp(0.0, 1.0)

        pos_mask = H_tgt.eq(1.0)
        neg_mask = H_tgt.lt(1.0)

        pos_loss = torch.pow(1.0 - H_pred, self.center_alpha) * torch.log(H_pred) * pos_mask.float()
        neg_loss = (
            torch.pow(1.0 - H_tgt, self.center_beta)
            * torch.pow(H_pred, self.center_alpha)
            * torch.log(1.0 - H_pred)
            * neg_mask.float()
        )

        num_pos = pos_mask.sum().float()
        if num_pos > 0:
            l_ctr = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            l_ctr = -neg_loss.sum() / max(1.0, float(b))
        l_ctr = torch.nan_to_num(l_ctr, nan=0.0, posinf=50.0, neginf=0.0)

        # 3. Spike-over-galaxy pixel upweighting
        w_pix = compute_spike_over_galaxy_weights(inst_gt, sem_gt, image=image, upweight_factor=3.0)

        # 4. Spike-Bridging Gate Loss (Weighted BCE with foreground ground truth sem_gt > 0)
        fg_target = (sem_gt > 0).unsqueeze(1).float()
        gate_bce = F.binary_cross_entropy(G_fg, fg_target, weight=w_pix, reduction="mean")
        l_gate = gate_bce

        # 5. Instance Query Matching Loss
        n_queries = M_inst.shape[1]
        l_inst_bce = torch.tensor(0.0, device=device)
        l_inst_dice = torch.tensor(0.0, device=device)
        matched_count = 0

        for i in range(b):
            gt_pts = batch_centroids[i]
            if len(gt_pts) == 0:
                continue

            inst_i = inst_gt[i]
            unique_ids = torch.unique(inst_i)
            unique_ids = unique_ids[unique_ids > 0]

            coords_i = star_coords[i]  # (N, 2) [x, y]
            assigned_queries = set()

            for k_idx, (gx, gy) in enumerate(gt_pts):
                inst_id = unique_ids[k_idx].item() if k_idx < len(unique_ids) else None
                if inst_id is None:
                    continue

                # Find nearest unassigned query
                dists = torch.sqrt((coords_i[:, 0] - gx) ** 2 + (coords_i[:, 1] - gy) ** 2)
                sorted_indices = torch.argsort(dists)

                best_q = None
                for q_idx in sorted_indices.tolist():
                    if q_idx not in assigned_queries:
                        best_q = q_idx
                        assigned_queries.add(q_idx)
                        break

                if best_q is None:
                    continue

                pred_m = M_inst[i, best_q]  # (H, W) in [0, 1]
                target_m = (inst_i == inst_id).float()
                w_m = w_pix[i, 0]

                # Weighted BCE
                bce = F.binary_cross_entropy(
                    pred_m.clamp(self.eps, 1.0 - self.eps), target_m, weight=w_m, reduction="mean"
                )

                # Soft Dice
                inter = (pred_m * target_m).sum()
                denom = pred_m.sum() + target_m.sum()
                dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)

                l_inst_bce = l_inst_bce + bce
                l_inst_dice = l_inst_dice + dice
                matched_count += 1

        if matched_count > 0:
            l_inst_bce = l_inst_bce / matched_count
            l_inst_dice = l_inst_dice / matched_count

        l_inst = l_inst_bce + l_inst_dice

        total_loss = (
            self.sem_weight * l_sem
            + self.center_weight * l_ctr
            + self.gate_weight * l_gate
            + self.inst_weight * l_inst
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=0.0)

        return {
            "loss": total_loss,
            "sem_loss": l_sem,
            "center_loss": l_ctr,
            "gate_loss": l_gate,
            "inst_loss": l_inst,
        }


def build_steer_query_optimizer(
    model: nn.Module,
    base_lr: float = 6e-5,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """
    Two-Speed Optimizer for SteerQuery:
    Assigns raw_dtheta and raw_perp_shifts geometric parameters to a separate
    parameter group with 0.1x learning rate and 0.0 weight decay.
    """
    steered_params = []
    regular_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "raw_dtheta" in name or "raw_perp_shifts" in name:
            steered_params.append(param)
        else:
            regular_params.append(param)

    param_groups = [
        {"params": regular_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": steered_params, "lr": base_lr * 0.1, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
