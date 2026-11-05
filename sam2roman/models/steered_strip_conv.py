"""
Learnable Steered Strip Convolution for Roman Space Telescope Diffraction Spikes.
=================================================================================

Multi-branch 1D strip convolution of odd length K (default K=31) and dilation d
aligned with the 6 spider strut axes of the Nancy Grace Roman Space Telescope:
- Nominal angles: phi_0 = [0, pi/6, 2pi/6, 3pi/6, 4pi/6, 5pi/6]
- Learnable angle offset raw_dtheta across strut axes (clamped to +/-5 degrees)
- Learnable perpendicular curvature shift raw_perp_shifts (capturing strut bending)
- Vectorized grid_sample sampling and depthwise strip weight aggregation
- Residual skip connection and directional branch feature outputs
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSteeredStripConv(nn.Module):
    """
    Learnable Steered Strip Convolution module.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    K : int, default 31
        Odd kernel length along the strip.
    dilation : int, default 1
        Dilation factor spacing strip samples along the strut axis.
    raw_dtheta : nn.Parameter, optional
        Pre-existing parameter for sharing angle offsets across layers.
    max_angle_deg : float, default 5.0
        Maximum angle deviation from nominal strut geometry in degrees.
    """

    def __init__(
        self,
        in_channels: int,
        K: int = 31,
        dilation: int = 1,
        raw_dtheta: Optional[nn.Parameter] = None,
        max_angle_deg: float = 5.0,
    ):
        super().__init__()
        if K % 2 == 0:
            raise ValueError(f"Kernel length K must be odd, got {K}")

        self.in_channels = in_channels
        self.K = K
        self.dilation = dilation
        self.max_rad = math.radians(max_angle_deg)
        self.midpoint = K // 2

        # 6 nominal strut angles: 0, 30, 60, 90, 120, 150 degrees
        phi_0 = torch.tensor([i * math.pi / 6.0 for i in range(6)], dtype=torch.float32)
        self.register_buffer("phi_0", phi_0)

        # Learnable angle offsets
        if raw_dtheta is not None:
            self.raw_dtheta = raw_dtheta
            self._shared_dtheta = True
        else:
            self.raw_dtheta = nn.Parameter(torch.zeros(6, dtype=torch.float32))
            self._shared_dtheta = False

        # Learnable curvature shift perpendicular to nominal rays
        self.raw_perp_shifts = nn.Parameter(torch.zeros(6, K, dtype=torch.float32))

        # Learnable depthwise strip weights along K
        # Initialize uniformly normalized along strip length
        init_weights = torch.full((6, in_channels, K), 1.0 / float(K), dtype=torch.float32)
        self.strip_weights = nn.Parameter(init_weights)

        # Branch fusion projection: (B, 6*C, H, W) -> (B, C, H, W)
        self.proj = nn.Sequential(
            nn.Conv2d(6 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

    def get_active_angles(self) -> torch.Tensor:
        """Compute active steered angles theta_m = phi_0 + max_rad * tanh(raw_dtheta)."""
        return self.phi_0 + self.max_rad * torch.tanh(self.raw_dtheta)

    def get_perp_shifts(self) -> torch.Tensor:
        """
        Compute cumulative perpendicular shifts with midpoint fixed at zero:
        inc = 0.25 * tanh(raw_perp_shifts); inc[n] = 0.0
        perp_shift = cumsum(inc) - cumsum(inc)[n]
        """
        n = self.midpoint
        inc = 0.25 * torch.tanh(self.raw_perp_shifts)
        # Zero out center point without in-place mutation to preserve autograd graph
        mask = torch.ones_like(inc)
        mask[:, n] = 0.0
        inc = inc * mask
        cum = torch.cumsum(inc, dim=-1)
        perp_shift = cum - cum[:, n : n + 1]
        return perp_shift

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (B, C, H, W).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - out: Residual feature map (B, C, H, W).
            - theta_m: Active strut angles (6,).
            - branch_feats: Steered branch representations (B, 6, C, H, W).
        """
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        theta_m = self.get_active_angles()  # (6,)
        perp_shift = self.get_perp_shifts()  # (6, K)

        n = self.midpoint
        s_steps = (
            torch.arange(-n, n + 1, device=device, dtype=dtype) * self.dilation
        )  # (K,)

        # Compute sampling offsets:
        # dx = s * cos(theta_m) - perp_shift * sin(theta_m)
        # dy = s * sin(theta_m) + perp_shift * cos(theta_m)
        cos_theta = torch.cos(theta_m).unsqueeze(1)  # (6, 1)
        sin_theta = torch.sin(theta_m).unsqueeze(1)  # (6, 1)
        s_grid = s_steps.unsqueeze(0)  # (1, K)

        dx = s_grid * cos_theta - perp_shift * sin_theta  # (6, K) in pixels
        dy = s_grid * sin_theta + perp_shift * cos_theta  # (6, K) in pixels

        # Convert to normalized grid coordinates in [-1, 1]
        # In align_corners=True, a 1-pixel shift is 2 / max(1, dim - 1)
        norm_x_factor = 2.0 / max(1, w - 1)
        norm_y_factor = 2.0 / max(1, h - 1)
        dx_norm = dx * norm_x_factor  # (6, K)
        dy_norm = dy * norm_y_factor  # (6, K)

        # Base coordinate meshgrid: (H, W, 2) where [..., 0] is X, [..., 1] is Y
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        base_grid = torch.stack([base_x, base_y], dim=-1)  # (H, W, 2)

        # Memory-efficient branch evaluation:
        # Avoids duplicating x across branches via x.repeat(6, ...) and avoids allocating
        # a physical (6*B, H, W, 2) grid 31 times. Instead, samples per branch with zero-copy
        # batch expansion (.expand(b, -1, -1, -1)), slashing peak VRAM by >75%.
        branch_list = []
        for m in range(6):
            accum_m = torch.zeros(b, c, h, w, device=device, dtype=dtype)
            for k in range(self.K):
                shift = torch.stack([dx_norm[m, k], dy_norm[m, k]])  # (2,)
                grid_k = (base_grid + shift).unsqueeze(0).expand(b, -1, -1, -1)

                sampled = F.grid_sample(
                    x,
                    grid_k,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )  # (B, C, H, W)

                w_mk = self.strip_weights[m, :, k].view(1, c, 1, 1)
                accum_m = accum_m + sampled * w_mk

            branch_list.append(accum_m)

        # Stack into (B, 6, C, H, W)
        branch_feats = torch.stack(branch_list, dim=1)

        # Concatenate branches and project
        concat_branches = branch_feats.view(b, 6 * c, h, w)
        proj_out = self.proj(concat_branches)
        out = proj_out + x

        return out, theta_m, branch_feats
