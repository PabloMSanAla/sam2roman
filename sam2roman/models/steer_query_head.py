"""
SteerQuery Instance Segmentation Head for Roman Space Telescope Astronomical Imagery.
====================================================================================

Directly predicts 2D integer instance segmentation masks on the GPU for deblending
stars with extensive diffraction spikes (100–4,000 px) crossing unmasked background galaxies.

Comprises:
1. SteeredPrototypeNeck: Multi-dilation strip-conv cascade with Plücker ray coordinate injection.
2. SpikeBridgingGate: Fuses semantic foreground logits with maximum directional strip energies.
3. StarCenterQuerySelector: Single-class star centroid heatmap and dynamic peak selection.
4. SteeredRayCrossAttentionDecoder: Cross-attends star queries to log-spaced ray tokens along strut axes.
5. Direct GPU Integer Mask Assembly: Einsum dynamic kernel assembly and argmax instance labeling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2roman.models.unetpp import ConvBlock, UNetPlusPlus
from sam2roman.models.steered_strip_conv import LearnableSteeredStripConv


def compute_fourier_features(
    coords: torch.Tensor,
    frequencies: Tuple[float, ...] = (1.0, 4.0, 16.0, 64.0),
) -> torch.Tensor:
    """
    Sinusoidal Fourier feature mapping for spatial coordinates.

    Parameters
    ----------
    coords : torch.Tensor
        Input coordinates of shape (B, C, H, W) or (B, N, C).
    frequencies : tuple of float
        Spatial frequency multipliers.

    Returns
    -------
    torch.Tensor:
        Fourier encoded features of shape (B, C * len(frequencies) * 2, ...).
    """
    device = coords.device
    dtype = coords.dtype
    freq_tensor = torch.tensor(frequencies, device=device, dtype=dtype) * math.pi

    if coords.ndim == 4:
        # (B, C, H, W)
        scaled = coords.unsqueeze(2) * freq_tensor.view(1, 1, -1, 1, 1)  # (B, C, F, H, W)
        sin_f = torch.sin(scaled)
        cos_f = torch.cos(scaled)
        fourier = torch.cat([sin_f, cos_f], dim=2)  # (B, C, 2*F, H, W)
        b, c, _, h, w = fourier.shape
        return fourier.view(b, c * len(frequencies) * 2, h, w)
    elif coords.ndim == 3:
        # (B, N, C)
        scaled = coords.unsqueeze(3) * freq_tensor.view(1, 1, 1, -1)  # (B, N, C, F)
        sin_f = torch.sin(scaled)
        cos_f = torch.cos(scaled)
        fourier = torch.cat([sin_f, cos_f], dim=3)  # (B, N, C, 2*F)
        b, n, c, _ = fourier.shape
        return fourier.view(b, n, c * len(frequencies) * 2)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got ndim={coords.ndim}")


class SteeredPrototypeNeck(nn.Module):
    """
    Steered Prototype Neck:
    - Shared-angle multi-dilation cascade (K=31, d=1, 4, 16)
    - Plücker ray coordinate injection: rho_m = gx*sin(theta) - gy*cos(theta), s_m = gx*cos(theta) + gy*sin(theta)
    - Sinusoidal Fourier encoding and 1x1 Conv projection to Mask Prototype Tensor P_proto (B, D, H, W).
    """

    def __init__(
        self,
        in_channels: int = 32,
        proto_dim: int = 32,
        K: int = 31,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.proto_dim = proto_dim
        self.K = K

        # Cascade of 3 steered strip conv layers sharing a single raw_dtheta parameter
        self.strip1 = LearnableSteeredStripConv(in_channels, K=K, dilation=1)
        self.strip2 = LearnableSteeredStripConv(
            in_channels, K=K, dilation=4, raw_dtheta=self.strip1.raw_dtheta
        )
        self.strip3 = LearnableSteeredStripConv(
            in_channels, K=K, dilation=16, raw_dtheta=self.strip1.raw_dtheta
        )

        # Plücker coordinates Fourier feature projection:
        # 12 Plücker channels * 4 frequencies * 2 (sin/cos) = 96 channels
        self.plucker_proj = nn.Sequential(
            nn.Conv2d(96, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # Prototype projection: (B, in_channels + in_channels, H, W) -> (B, proto_dim, H, W)
        self.proto_proj = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, proto_dim, kernel_size=1),
        )

    def forward(
        self, F_dec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        F_dec : torch.Tensor
            Decoder feature map of shape (B, C, H, W).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - P_proto: Mask prototype tensor of shape (B, proto_dim, H, W).
            - theta_m: Active strut angles of shape (6,).
            - branch_feats1: Directional features from 1st strip layer of shape (B, 6, C, H, W).
        """
        b, c, h, w = F_dec.shape
        device = F_dec.device
        dtype = F_dec.dtype

        # 1. Multi-dilation cascade
        x1, theta_m, branch_feats1 = self.strip1(F_dec)
        x2, _, _ = self.strip2(x1)
        x3, _, _ = self.strip3(x2)

        # 2. Normalized pixel coordinate grid in [-1, 1]
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )  # (H, W)

        gx_exp = gx.unsqueeze(0).expand(6, -1, -1)  # (6, H, W)
        gy_exp = gy.unsqueeze(0).expand(6, -1, -1)  # (6, H, W)

        sin_theta = torch.sin(theta_m).view(6, 1, 1)
        cos_theta = torch.cos(theta_m).view(6, 1, 1)

        # Plücker orthogonal offset rho_m and parallel coordinate s_m
        rho_m = gx_exp * sin_theta - gy_exp * cos_theta  # (6, H, W)
        s_m = gx_exp * cos_theta + gy_exp * sin_theta    # (6, H, W)
        plucker_12 = torch.cat([rho_m, s_m], dim=0).unsqueeze(0).expand(b, -1, -1, -1)  # (B, 12, H, W)

        # 3. Fourier encoding
        fourier_plucker = compute_fourier_features(plucker_12)  # (B, 96, H, W)
        plucker_feats = self.plucker_proj(fourier_plucker)       # (B, C, H, W)

        # 4. Prototype projection
        neck_fusion = torch.cat([x3, plucker_feats], dim=1)  # (B, 2*C, H, W)
        P_proto = self.proto_proj(neck_fusion)               # (B, proto_dim, H, W)

        return P_proto, theta_m, branch_feats1


class SpikeBridgingGate(nn.Module):
    """
    Steered Spike-Bridging Gate (G_fg):
    Fuses semantic foreground logits (classes 1: interior and 2: edge)
    with the maximum directional strip convolution energy across all 6 strut branches.
    """

    def __init__(self, in_channels: int = 32):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 + in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, logits_sem: torch.Tensor, branch_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits_sem : torch.Tensor
            3-class semantic logits (B, 3, H, W).
        branch_feats : torch.Tensor
            Directional branch features (B, 6, C, H, W).

        Returns
        -------
        torch.Tensor:
            Gate G_fg of shape (B, 1, H, W) with values in [0, 1].
        """
        # Classes 1 (interior) and 2 (edge)
        sem_fg = logits_sem[:, 1:3]  # (B, 2, H, W)
        branch_max = branch_feats.max(dim=1).values  # (B, C, H, W)

        gate_in = torch.cat([sem_fg, branch_max], dim=1)
        G_fg = self.gate_conv(gate_in)
        return G_fg


class StarCenterQuerySelector(nn.Module):
    """
    Star Center Query Selector:
    - Predicts single-class star center heatmap H_star and dense query embedding E_query.
    - Extracts candidate star coordinates via 3x3 max-pooling NMS.
    - Selects top-N queries (up to N_max) or accepts teacher-forced ground-truth centroids.
    """

    def __init__(
        self,
        in_channels: int = 32,
        query_dim: int = 32,
        N_max: int = 128,
        tau_star: float = 0.30,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.query_dim = query_dim
        self.N_max = N_max
        self.tau_star = tau_star

        # Star Center Heatmap Head: conditioning on 6 directional branch energies
        # suppresses false detections on galaxy bulges/clumps lacking 6-fold spikes.
        self.center_head = nn.Sequential(
            nn.Conv2d(in_channels + 6, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

        # Dense Query Embedding Head
        self.query_embed_head = nn.Sequential(
            nn.Conv2d(in_channels + 6, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, query_dim, kernel_size=1),
        )

        # CenterNet focal prior initialization
        if hasattr(self.center_head[-1], "bias") and self.center_head[-1].bias is not None:
            nn.init.constant_(self.center_head[-1].bias, -4.0)

    def forward(
        self,
        F_dec: torch.Tensor,
        branch_feats: torch.Tensor,
        gt_centroids: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        F_dec : torch.Tensor
            (B, C, H, W)
        branch_feats : torch.Tensor
            (B, 6, C, H, W)
        gt_centroids : list of list of (cx, cy), optional
            Ground truth star centers for teacher forcing during warmup.

        Returns
        -------
        Tuple containing:
            - H_star: Heatmap probability (B, 1, H, W)
            - E_query: Dense embedding map (B, query_dim, H, W)
            - star_coords: Query coordinates in pixels (B, N, 2) [x, y]
            - scores: Detection scores (B, N)
            - Q_0: Initial queries sampled from E_query (B, N, query_dim)
            - valid_mask: Boolean indicator for active queries (B, N)
        """
        b, c, h, w = F_dec.shape
        device = F_dec.device
        dtype = F_dec.dtype

        # Directional branch energy across channels: (B, 6, H, W)
        branch_energy = branch_feats.norm(p=2, dim=2)
        fused = torch.cat([F_dec, branch_energy], dim=1)  # (B, C + 6, H, W)

        center_logits = self.center_head(fused)
        H_star = torch.sigmoid(center_logits)
        E_query = self.query_embed_head(fused)

        coords_list = []
        scores_list = []
        valid_list = []

        # Peak extraction via 3x3 max-pooling NMS
        nms_mask = (H_star == F.max_pool2d(H_star, kernel_size=3, stride=1, padding=1)) & (H_star > 1e-4)

        for i in range(b):
            c_img = H_star[i, 0]
            # Teacher-forcing override if provided
            if gt_centroids is not None and len(gt_centroids) > i and len(gt_centroids[i]) > 0:
                pts = gt_centroids[i][: self.N_max]
                cx_t = torch.tensor([p[0] for p in pts], device=device, dtype=dtype)
                cy_t = torch.tensor([p[1] for p in pts], device=device, dtype=dtype)
                # Add 1-px Gaussian jitter during training teacher-forcing
                if self.training:
                    cx_t = cx_t + torch.randn_like(cx_t) * 0.5
                    cy_t = cy_t + torch.randn_like(cy_t) * 0.5
                cx_t = cx_t.clamp(0, w - 1)
                cy_t = cy_t.clamp(0, h - 1)

                sc_vals = c_img[cy_t.long(), cx_t.long()]
                k_valid = len(pts)
                xy = torch.stack([cx_t, cy_t], dim=-1)
                val_mask = torch.ones(k_valid, device=device, dtype=torch.bool)
            else:
                peak_mask = nms_mask[i, 0]
                ys, xs = torch.where(peak_mask)
                if len(ys) > 0:
                    peak_scores = c_img[ys, xs]
                    # Filter by tau_star during evaluation
                    if not self.training:
                        keep = peak_scores >= self.tau_star
                        ys = ys[keep]
                        xs = xs[keep]
                        peak_scores = peak_scores[keep]

                    if len(peak_scores) > 0:
                        topk = min(self.N_max, len(peak_scores))
                        top_vals, top_idx = torch.topk(peak_scores, topk)
                        xy = torch.stack([xs[top_idx].float(), ys[top_idx].float()], dim=-1)
                        sc_vals = top_vals
                        k_valid = topk
                        val_mask = torch.ones(k_valid, device=device, dtype=torch.bool)
                    else:
                        xy = torch.zeros(0, 2, device=device, dtype=dtype)
                        sc_vals = torch.zeros(0, device=device, dtype=dtype)
                        k_valid = 0
                        val_mask = torch.zeros(0, device=device, dtype=torch.bool)
                else:
                    xy = torch.zeros(0, 2, device=device, dtype=dtype)
                    sc_vals = torch.zeros(0, device=device, dtype=dtype)
                    k_valid = 0
                    val_mask = torch.zeros(0, device=device, dtype=torch.bool)

            # Pad up to N_max
            pad_count = self.N_max - k_valid
            if pad_count > 0:
                pad_xy = torch.zeros(pad_count, 2, device=device, dtype=dtype)
                pad_sc = torch.zeros(pad_count, device=device, dtype=dtype)
                pad_val = torch.zeros(pad_count, device=device, dtype=torch.bool)
                xy = torch.cat([xy, pad_xy], dim=0)
                sc_vals = torch.cat([sc_vals, pad_sc], dim=0)
                val_mask = torch.cat([val_mask, pad_val], dim=0)

            coords_list.append(xy)
            scores_list.append(sc_vals)
            valid_list.append(val_mask)

        star_coords = torch.stack(coords_list, dim=0)  # (B, N, 2)
        scores = torch.stack(scores_list, dim=0)        # (B, N)
        valid_mask = torch.stack(valid_list, dim=0)    # (B, N)

        # Sample Q_0 from E_query at star_coords
        norm_x = (2.0 * star_coords[..., 0] / max(1, w - 1)) - 1.0
        norm_y = (2.0 * star_coords[..., 1] / max(1, h - 1)) - 1.0
        query_grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)

        sampled_q = F.grid_sample(
            E_query,
            query_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, query_dim, N, 1)

        Q_0 = sampled_q.squeeze(-1).permute(0, 2, 1).contiguous()  # (B, N, query_dim)

        return H_star, E_query, star_coords, scores, Q_0, valid_mask


class SteeredRayCrossAttentionDecoder(nn.Module):
    """
    12-Ray Cross-Attention Query Decoder:
    - Samples E_query along 12 learned strut rays (theta_m and theta_m + pi)
      at log-spaced radial steps r in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512].
    - Injects Plücker sinusoidal ray equation of query centers.
    - 2-layer Cross-Attention + FFN decodes dynamic instance filter weights W_mask (B, N, D).
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.radial_steps = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        self.num_rays = 12
        self.num_radial = len(self.radial_steps)
        self.num_tokens = self.num_rays * self.num_radial  # 120 tokens

        # Plücker query coordinate Fourier projection (12 * 8 = 96 -> embed_dim)
        self.plucker_query_proj = nn.Sequential(
            nn.Linear(96, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # 2-layer Cross-Attention Decoder
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "cross_attn": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                    "norm1": nn.LayerNorm(embed_dim),
                    "ffn": nn.Sequential(
                        nn.Linear(embed_dim, embed_dim * 2),
                        nn.GELU(),
                        nn.Linear(embed_dim * 2, embed_dim),
                    ),
                    "norm2": nn.LayerNorm(embed_dim),
                })
            )

        # Output MLP predicting dynamic filter weights
        self.mask_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        Q_0: torch.Tensor,
        E_query: torch.Tensor,
        star_coords: torch.Tensor,
        theta_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        Q_0 : torch.Tensor
            Initial queries (B, N, D).
        E_query : torch.Tensor
            Dense query embedding map (B, D, H, W).
        star_coords : torch.Tensor
            Query coordinates in pixels [x, y] of shape (B, N, 2).
        theta_m : torch.Tensor
            Active strut angles of shape (6,).

        Returns
        -------
        torch.Tensor:
            Dynamic mask filter weights W_mask of shape (B, N, D).
        """
        b, n, d = Q_0.shape
        _, _, h, w = E_query.shape
        device = Q_0.device
        dtype = Q_0.dtype

        # 12 ray angles: theta_m and theta_m + pi
        ray_angles = torch.cat([theta_m, theta_m + math.pi], dim=0)  # (12,)
        r_tensor = torch.tensor(self.radial_steps, device=device, dtype=dtype)  # (10,)

        # Radial offsets: (12, 10)
        cos_rays = torch.cos(ray_angles).view(12, 1)
        sin_rays = torch.sin(ray_angles).view(12, 1)
        r_grid = r_tensor.view(1, 10)

        delta_x = (cos_rays * r_grid).reshape(1, 1, 120)  # (1, 1, 120)
        delta_y = (sin_rays * r_grid).reshape(1, 1, 120)  # (1, 1, 120)

        # Query points along rays: (B, N, 120, 2) in pixels
        qx = star_coords[..., 0:1] + delta_x  # (B, N, 120)
        qy = star_coords[..., 1:2] + delta_y  # (B, N, 120)

        # Convert to normalized grid in [-1, 1]
        norm_qx = (2.0 * qx / max(1, w - 1)) - 1.0
        norm_qy = (2.0 * qy / max(1, h - 1)) - 1.0
        ray_grid = torch.stack([norm_qx, norm_qy], dim=-1)  # (B, N, 120, 2)

        # Sample ray tokens: grid shape (B, N*120, 1, 2)
        ray_grid_flat = ray_grid.view(b, n * 120, 1, 2)
        sampled_rays = F.grid_sample(
            E_query,
            ray_grid_flat,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, D, N*120, 1)

        ray_tokens = (
            sampled_rays.squeeze(-1)
            .permute(0, 2, 1)
            .contiguous()
            .view(b * n, 120, d)
        )  # (B*N, 120, D)

        # Plücker coordinates of query centroids
        gx = (2.0 * star_coords[..., 0] / max(1, w - 1)) - 1.0  # (B, N)
        gy = (2.0 * star_coords[..., 1] / max(1, h - 1)) - 1.0  # (B, N)

        # Compute Plücker coordinates for all 6 strut axes
        sin_th = torch.sin(theta_m).view(1, 1, 6)
        cos_th = torch.cos(theta_m).view(1, 1, 6)
        gx_6 = gx.unsqueeze(-1)  # (B, N, 1)
        gy_6 = gy.unsqueeze(-1)  # (B, N, 1)

        rho = gx_6 * sin_th - gy_6 * cos_th  # (B, N, 6)
        s = gx_6 * cos_th + gy_6 * sin_th    # (B, N, 6)
        plucker_q = torch.cat([rho, s], dim=-1)  # (B, N, 12)

        # Fourier encoding of query Plücker ray parameters
        fourier_q = compute_fourier_features(plucker_q)  # (B, N, 96)
        plucker_embed = self.plucker_query_proj(fourier_q)  # (B, N, D)

        # Query state
        Q = (Q_0 + plucker_embed).view(b * n, 1, d)  # (B*N, 1, D)

        # Cross-Attention over ray tokens
        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](Q, ray_tokens, ray_tokens)
            Q = layer["norm1"](Q + attn_out)
            ffn_out = layer["ffn"](Q)
            Q = layer["norm2"](Q + ffn_out)

        Q_out = Q.view(b, n, d)
        W_mask = self.mask_mlp(Q_out)  # (B, N, D)
        return W_mask


class SteerQueryHead(nn.Module):
    """
    SteerQuery Instance Segmentation Head.

    Combines:
    - SteeredPrototypeNeck (Plücker injection + multi-dilation strip convs)
    - SpikeBridgingGate (preserving spike continuity across background galaxies)
    - StarCenterQuerySelector (repressing galaxy bulges via 6-fold symmetry)
    - SteeredRayCrossAttentionDecoder (decoding dynamic filters from ray tokens)
    - Direct GPU integer instance mask assembly (torch.int32)
    """

    def __init__(
        self,
        in_channels: int = 32,
        proto_dim: int = 32,
        N_max: int = 128,
        tau_mask: float = 0.35,
        tau_star: float = 0.30,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.proto_dim = proto_dim
        self.N_max = N_max
        self.tau_mask = tau_mask
        self.tau_star = tau_star

        self.proto_neck = SteeredPrototypeNeck(in_channels=in_channels, proto_dim=proto_dim)
        self.bridging_gate = SpikeBridgingGate(in_channels=in_channels)
        self.query_selector = StarCenterQuerySelector(
            in_channels=in_channels, query_dim=proto_dim, N_max=N_max, tau_star=tau_star
        )
        self.ray_decoder = SteeredRayCrossAttentionDecoder(embed_dim=proto_dim)

    def forward(
        self,
        F_dec: torch.Tensor,
        logits_sem: torch.Tensor,
        gt_centroids: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        F_dec : torch.Tensor
            Final high-resolution decoder feature map (B, C_dec, H, W).
        logits_sem : torch.Tensor
            3-class semantic segmentation logits (B, 3, H, W).
        gt_centroids : list of list of (cx, cy), optional
            Ground truth star centers for teacher forcing during warmup.

        Returns
        -------
        Dict[str, Any]:
            - 'instance_mask': (B, H, W) torch.int32 integer mask (0: bg/galaxies, 1..N: stars).
            - 'logits_sem': (B, 3, H, W) unmodified semantic logits.
            - 'H_star': (B, 1, H, W) star center detection probability heatmap.
            - 'M_inst': (B, N, H, W) individual instance mask probabilities.
            - 'G_fg': (B, 1, H, W) spike bridging gate.
            - 'theta_m': (6,) active strut angles.
            - 'scores': (B, N) detection confidence scores.
            - 'star_coords': (B, N, 2) detected star centers.
            - 'valid_mask': (B, N) indicator of active query instances.
        """
        b, _, h, w = F_dec.shape
        device = F_dec.device
        dtype = F_dec.dtype

        # 1. Prototype Neck & Steered Active Angles
        P_proto, theta_m, branch_feats = self.proto_neck(F_dec)

        # 2. Spike-Bridging Gate
        G_fg = self.bridging_gate(logits_sem, branch_feats)

        # 3. Star Center Query Selection
        H_star, E_query, star_coords, scores, Q_0, valid_mask = self.query_selector(
            F_dec, branch_feats, gt_centroids=gt_centroids
        )

        # 4. 12-Ray Cross-Attention Query Decoder
        W_mask = self.ray_decoder(Q_0, E_query, star_coords, theta_m)  # (B, N, D)

        # 5. Direct GPU Integer Mask Assembly
        # Instance mask logits: einsum('bnd,bdhw->bnhw')
        mask_logits = torch.einsum("bnd,bdhw->bnhw", W_mask, P_proto)

        # Apply spike-bridging gate
        M_inst = torch.sigmoid(mask_logits) * G_fg  # (B, N, H, W)

        # Zero out invalid padded queries
        M_inst = M_inst * valid_mask.view(b, self.N_max, 1, 1).float()

        # Weight by detection confidence
        M_scored = M_inst * scores.view(b, self.N_max, 1, 1)

        # Prepend background channel 0 filled with threshold tau_mask
        bg = torch.full((b, 1, h, w), self.tau_mask, device=device, dtype=dtype)
        scored_with_bg = torch.cat([bg, M_scored], dim=1)  # (B, N + 1, H, W)

        # Argmax assembly: 0 is background / galaxy, 1..N are unique star instances
        instance_mask = torch.argmax(scored_with_bg, dim=1).to(torch.int32)

        return {
            "instance_mask": instance_mask,
            "logits_sem": logits_sem,
            "H_star": H_star,
            "M_inst": M_inst,
            "G_fg": G_fg,
            "theta_m": theta_m,
            "scores": scores,
            "star_coords": star_coords,
            "valid_mask": valid_mask,
        }


class UNetPlusPlusSteerQuery(UNetPlusPlus):
    """
    Complete Star-Deblending Network combining the unmodified UNet++ backbone
    with the SteerQuery instance segmentation head.

    Maintains identical layer names to UNetPlusPlus (conv0_0..conv0_4, final)
    so existing pretrained UNet++ checkpoints load cleanly via load_state_dict(..., strict=False).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256, 512),
        norm_type: str = "batch",
        proto_dim: int = 32,
        N_max: int = 128,
        tau_mask: float = 0.35,
        tau_star: float = 0.30,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            deep_supervision=False,
            features=features,
            norm_type=norm_type,
        )

        c_dec = features[0]  # Final decoder channels (nb[0] = 32)
        self.steer_query_head = SteerQueryHead(
            in_channels=c_dec,
            proto_dim=proto_dim,
            N_max=N_max,
            tau_mask=tau_mask,
            tau_star=tau_star,
        )

    def forward(
        self,
        x: torch.Tensor,
        gt_centroids: Optional[List[List[Tuple[float, float]]]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, in_channels, H, W).
        gt_centroids : list of list of (cx, cy), optional
            Ground truth star centers for teacher forcing during warmup.

        Returns
        -------
        Dict[str, Any]:
            Dictionary containing 'instance_mask', 'logits_sem', 'H_star', 'M_inst', 'G_fg', etc.
        """
        # Column 0: Backbone Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column 4 (Node X^{0,4})
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # 3-class semantic logits
        logits_sem = self.final(x0_4)

        # Instance segmentation head
        return self.steer_query_head(x0_4, logits_sem, gt_centroids=gt_centroids)
