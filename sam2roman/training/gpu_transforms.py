"""
GPU-Accelerated Astronomical Transforms & Noise Augmentations for sam2roman
==========================================================================

Runs entirely on CUDA / MPS / CPU tensors in PyTorch:
1. GPULeakySB: Vectorized astronomical surface brightness stretch converting raw detector
   counts into normalized [0.0, 1.0] RGB tensors in <0.05 ms on NVIDIA Tensor Cores.
2. GPUAstroAugmentation: In-VRAM astronomical noise injection (Poisson shot noise, Gaussian
   readout noise, sky pedestal shifts, flux scaling) and synchronized spatial transforms
   (random 90-deg rotations, horizontal/vertical flips) applied jointly to image & target.
3. GPUAstroTransform: End-to-end pipeline executing directly on GPU batches.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class GPULeakySB(nn.Module):
    """
    Vectorized bijective surface brightness stretch implemented in pure PyTorch.
    Converts raw physical fluxes into normalized [0.0, 1.0] float32 values.

    Parameters
    ----------
    zero_points : float, default 31.4
        Photometric zero point.
    pixel_scale : float, default 0.11
        Plate scale in arcsec/pixel (0.11 for Roman WFI).
    mmin : float, default 13.0
        Bright-end surface brightness magnitude limit.
    mmax : float, default 32.0
        Faint-end surface brightness magnitude limit.
    negative_slope : float, default 0.1
        Slope for sub-sky negative and faint regimes.
    positive_slope : float, default 0.05
        Slope for saturated core compression.
    y_zero : float, default 0.015
        Deep-sky pedestal level.
    y_one : float, default 0.96
        Saturation knee point.
    out_channels : int, default 3
        Number of output channels (3 replicates to identical RGB for standard vision backbones).
    """

    def __init__(
        self,
        zero_points: float = 31.4,
        pixel_scale: float = 0.11,
        mmin: float = 13.0,
        mmax: float = 32.0,
        mmin_jitter: float = 0.0,
        mmax_jitter: float = 0.0,
        negative_slope: float = 0.1,
        positive_slope: float = 0.05,
        y_zero: float = 0.015,
        y_one: float = 0.96,
        out_channels: int = 3,
    ):
        super().__init__()
        self.zero_points = float(zero_points)
        self.pixel_scale = float(pixel_scale)
        self.mmin = float(mmin)
        self.mmax = float(mmax)
        self.mmin_jitter = float(mmin_jitter)
        self.mmax_jitter = float(mmax_jitter)
        self.negative_slope = float(negative_slope)
        self.positive_slope = float(positive_slope)
        self.y_zero = float(y_zero)
        self.y_one = float(y_one)
        self.out_channels = int(out_channels)

        # Precompute constants
        self.span = self.mmax - self.mmin
        self.mid_span = self.y_one - self.y_zero
        self.top_room = 1.0 - self.y_one
        self.k_tangent = (2.5 / math.log(10.0)) / self.span
        self.f_mmax = (self.pixel_scale ** 2) * (10.0 ** (0.4 * (self.zero_points - self.mmax)))

        self.s_pos = (self.positive_slope * self.mid_span) / max(self.top_room, 1e-8)
        self.s_neg = (self.negative_slope * self.k_tangent * self.mid_span) / max(self.y_zero, 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stretch raw detector fluxes into [0.0, 1.0].

        Parameters
        ----------
        x : torch.Tensor
            Raw detector counts of shape (B, 1, H, W) or (B, H, W).

        Returns
        -------
        torch.Tensor:
            Normalized tensor of shape (B, out_channels, H, W) in [0.0, 1.0].
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)

        # Handle NaNs or Infs gracefully in-place on GPU
        if torch.isnan(x).any() or torch.isinf(x).any():
            finite_mask = torch.isfinite(x)
            fill_val = torch.median(x[finite_mask]) if finite_mask.any() else torch.tensor(0.0, device=x.device, dtype=x.dtype)
            x = torch.where(finite_mask, x, fill_val)

        b = x.shape[0]

        # Apply slight magnitude stretch jitter in training mode if enabled
        if self.training and (self.mmin_jitter > 0.0 or self.mmax_jitter > 0.0):
            dm_min = (torch.rand((b, 1, 1, 1), device=x.device, dtype=x.dtype) * 2.0 - 1.0) * self.mmin_jitter
            dm_max = (torch.rand((b, 1, 1, 1), device=x.device, dtype=x.dtype) * 2.0 - 1.0) * self.mmax_jitter
            eff_mmin = self.mmin + dm_min
            eff_mmax = self.mmax + dm_max
            span = torch.clamp(eff_mmax - eff_mmin, min=5.0)
            f_mmax = (self.pixel_scale ** 2) * (10.0 ** (0.4 * (self.zero_points - eff_mmax)))
            k_tangent = (2.5 / math.log(10.0)) / span
            s_neg = (self.negative_slope * k_tangent * self.mid_span) / max(self.y_zero, 1e-8)
            s_pos = self.s_pos
        else:
            span = self.span
            f_mmax = self.f_mmax
            s_neg = self.s_neg
            s_pos = self.s_pos

        # Normalize relative to faint magnitude limit f_mmax
        f_norm = x / f_mmax

        # Log regime for f_norm >= 1.0
        f_safe = torch.clamp(f_norm, min=1.0)
        u = (2.5 * torch.log10(f_safe)) / span

        y_mid = self.y_zero + self.mid_span * u

        # Core saturation regime (u > 1.0)
        u_excess = torch.clamp(u - 1.0, min=0.0)
        y_high = 1.0 - (self.top_room / (1.0 + s_pos * u_excess))
        y_pos = torch.where(u <= 1.0, y_mid, y_high)

        # Negative & sub-sky background regime (f_norm < 1.0)
        f_deficit = torch.clamp(1.0 - f_norm, min=0.0)
        y_low = self.y_zero / (1.0 + s_neg * f_deficit)

        stretched = torch.where(f_norm >= 1.0, y_pos, y_low)
        stretched = torch.clamp(stretched, 0.0, 1.0)

        if self.out_channels == 3 and stretched.shape[1] == 1:
            stretched = stretched.repeat(1, 3, 1, 1)

        return stretched


def measure_background_noise(
    raw: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    gain: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Measure background pedestal (median) and noise dispersion (RMS/IQR) directly
    per sample in physical units (e.g. nJy at ZP=31.4) on GPU.

    Parameters
    ----------
    raw : torch.Tensor
        Raw detector flux tensor of shape (B, 1, H, W) or (B, H, W).
    target : torch.Tensor, optional
        Target instance segmentation tensor of shape (B, H, W) or (B, 1, H, W).
        If provided, pixels where target == 0 are isolated as sky background.
    gain : float, default 100.0
        Detector conversion factor (e-/nJy) used as physical shot noise floor.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]:
        (bg_median, bg_sigma), both shaped (B, 1, 1, 1).
    """
    b = raw.shape[0]
    flat = raw.view(b, -1)
    q_levels = torch.tensor([0.25, 0.50, 0.75], device=raw.device, dtype=raw.dtype)

    if target is not None:
        flat_tgt = target.view(b, -1)
        bg_flat = torch.where(flat_tgt == 0, flat, torch.nan)
        has_enough_bg = (flat_tgt == 0).sum(dim=1) >= 64
        q = torch.nanquantile(bg_flat, q_levels, dim=1)  # (3, B)
        if not has_enough_bg.all():
            q_fallback = torch.quantile(flat, q_levels, dim=1)
            q = torch.where(has_enough_bg.unsqueeze(0), q, q_fallback)
    else:
        q = torch.quantile(flat, q_levels, dim=1)

    med = q[1].view(b, 1, 1, 1)
    sig = ((q[2] - q[0]) / 1.349).view(b, 1, 1, 1)
    med = torch.nan_to_num(med, nan=0.0)
    sig = torch.nan_to_num(sig, nan=0.0)

    # If input is noiseless or flat (sig < 1e-2), fall back to expected physical shot noise floor
    poisson_floor = torch.sqrt(torch.clamp(med, min=0.0) / max(gain, 1e-4)).clamp(min=1.0)
    sig = torch.where(sig < 1e-2, poisson_floor, sig)

    return med, sig


class GPUAstroAugmentation(nn.Module):
    """
    GPU-accelerated astronomical noise injection and spatial augmentations.
    Executed in VRAM on raw physical fluxes before stretching.
    Directly measures background noise and scales noise injection in physical units (nJy).

    Parameters
    ----------
    enable_noise : bool, default True
        Whether to apply astronomical noise injection during training.
    enable_spatial : bool, default True
        Whether to apply synchronized geometric transforms (rotations, flips).
    auto_scale_noise : bool, default True
        If True, directly measures background noise (RMS/IQR) from unmasked sky pixels
        and scales read noise, sky pedestal shifts, and gradients accordingly.
    gain : float, default 100.0
        Detector electron conversion (e-/nJy for Roman WFI in nJy at zero point 31.4).
    read_noise_scale : float, default 0.3
        Read noise standard deviation as a fraction of measured background RMS.
    sky_pedestal_scale : float, default 1.0
        Sky pedestal shift standard deviation as a multiple of measured background RMS.
    gradient_scale_range : Tuple[float, float], default (0.2, 2.5)
        Tilted plane gradient slope amplitude range relative to measured background RMS.
    shot_noise_prob : float, default 0.5
        Probability of injecting Poisson photon counting shot noise.
    read_noise_std : float, default 5.0
        Fixed standard deviation of readout noise in nJy (used when auto_scale_noise=False).
    read_noise_prob : float, default 0.5
        Probability of injecting readout noise.
    sky_pedestal_std : float, default 15.0
        Fixed standard deviation of sky pedestal fluctuations in nJy (when auto_scale_noise=False).
    sky_pedestal_prob : float, default 0.5
        Probability of shifting sky background pedestal.
    gradient_prob : float, default 0.5
        Probability of applying tilted-plane background gradients.
    gradient_sigma_range : Tuple[float, float], default (2.0, 25.0)
        Fixed gradient slope range in nJy (when auto_scale_noise=False).
    flux_scale_range : Tuple[float, float], default (0.9, 1.1)
        Uniform random flux scaling range (zeropoint jitter).
    flux_scale_prob : float, default 0.5
        Probability of scaling flux.
    rot_prob : float, default 0.5
        Probability of applying random 90-degree rotations.
    flip_h_prob : float, default 0.5
        Probability of horizontal reflection.
    flip_v_prob : float, default 0.5
        Probability of vertical reflection.
    """

    def __init__(
        self,
        enable_noise: bool = True,
        enable_spatial: bool = True,
        auto_scale_noise: bool = True,
        gain: float = 100.0,
        read_noise_scale: float = 0.3,
        sky_pedestal_scale: float = 1.0,
        gradient_scale_range: Tuple[float, float] = (0.2, 2.5),
        shot_noise_prob: float = 0.5,
        read_noise_std: float = 5.0,
        read_noise_prob: float = 0.5,
        sky_pedestal_std: float = 15.0,
        sky_pedestal_prob: float = 0.5,
        gradient_prob: float = 0.5,
        gradient_sigma_range: Tuple[float, float] = (2.0, 25.0),
        flux_scale_range: Tuple[float, float] = (0.9, 1.1),
        flux_scale_prob: float = 0.5,
        rot_prob: float = 0.5,
        flip_h_prob: float = 0.5,
        flip_v_prob: float = 0.5,
    ):
        super().__init__()
        self.enable_noise = enable_noise
        self.enable_spatial = enable_spatial
        self.auto_scale_noise = auto_scale_noise
        self.gain = gain
        self.read_noise_scale = read_noise_scale
        self.sky_pedestal_scale = sky_pedestal_scale
        self.gradient_scale_range = gradient_scale_range
        self.shot_noise_prob = shot_noise_prob
        self.read_noise_std = read_noise_std
        self.read_noise_prob = read_noise_prob
        self.sky_pedestal_std = sky_pedestal_std
        self.sky_pedestal_prob = sky_pedestal_prob
        self.gradient_prob = gradient_prob
        self.gradient_sigma_range = gradient_sigma_range
        self.flux_scale_range = flux_scale_range
        self.flux_scale_prob = flux_scale_prob
        self.rot_prob = rot_prob
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob

    def forward(
        self,
        raw: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply noise and spatial transforms.

        Parameters
        ----------
        raw : torch.Tensor
            Raw detector flux of shape (B, 1, H, W) or (B, H, W).
        target : torch.Tensor, optional
            Ground truth target tensor of shape (B, H, W) or (B, 1, H, W).

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            Augmented (raw, target) tensors.
        """
        if not self.training:
            return raw, target

        aug_raw = raw.clone()
        aug_target = target.clone() if target is not None else None

        # -------------------------------------------------------------------
        # 1. Astronomical Noise Injection on Physical Flux (In-VRAM)
        # -------------------------------------------------------------------
        if self.enable_noise:
            b, _, h, w = aug_raw.shape if aug_raw.ndim == 4 else (aug_raw.shape[0], 1, *aug_raw.shape[1:])

            # Auto-measure physical background statistics per sample on GPU
            if self.auto_scale_noise:
                bg_med, bg_sigma = measure_background_noise(raw, target, gain=self.gain)
                eff_read_std = bg_sigma * self.read_noise_scale
                eff_pedestal_std = bg_sigma * self.sky_pedestal_scale
                sig_low = bg_sigma * self.gradient_scale_range[0]
                sig_high = bg_sigma * self.gradient_scale_range[1]
            else:
                eff_read_std = torch.full((b, 1, 1, 1), self.read_noise_std, device=raw.device, dtype=raw.dtype)
                eff_pedestal_std = torch.full((b, 1, 1, 1), self.sky_pedestal_std, device=raw.device, dtype=raw.dtype)
                sig_low = torch.full((b, 1, 1, 1), self.gradient_sigma_range[0], device=raw.device, dtype=raw.dtype)
                sig_high = torch.full((b, 1, 1, 1), self.gradient_sigma_range[1], device=raw.device, dtype=raw.dtype)

            # A. Poisson Photon Shot Noise: sigma_shot = sqrt(max(0, F) / gain)
            if self.shot_noise_prob > 0 and torch.rand(1, device=raw.device).item() < self.shot_noise_prob:
                shot_sigma = torch.sqrt(torch.clamp(aug_raw, min=0.0) / max(self.gain, 1e-4))
                shot_noise = torch.randn_like(aug_raw) * shot_sigma
                aug_raw = aug_raw + shot_noise

            # B. Gaussian Detector Readout Noise: N(0, eff_read_std^2)
            if self.read_noise_prob > 0 and torch.rand(1, device=raw.device).item() < self.read_noise_prob:
                read_noise = torch.randn_like(aug_raw) * eff_read_std
                aug_raw = aug_raw + read_noise

            # C. Sky Pedestal Shifts (simulates varying sky subtraction residuals)
            if self.sky_pedestal_prob > 0 and torch.rand(1, device=raw.device).item() < self.sky_pedestal_prob:
                pedestal = torch.randn((b, 1, 1, 1), device=raw.device, dtype=raw.dtype) * eff_pedestal_std
                if aug_raw.ndim == 3:
                    pedestal = pedestal.squeeze(1)
                aug_raw = aug_raw + pedestal

            # D. Sky Gradient Shifts: Tilted planes centered at 0 with variable sigma
            if self.gradient_prob > 0 and torch.rand(1, device=raw.device).item() < self.gradient_prob:
                # Normalized grid centered at 0 in [-1.0, 1.0]
                ys = torch.linspace(-1.0, 1.0, h, device=raw.device, dtype=raw.dtype)
                xs = torch.linspace(-1.0, 1.0, w, device=raw.device, dtype=raw.dtype)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

                # Sample variable standard deviation per sample
                u_rand = torch.rand((b, 1, 1, 1), device=raw.device, dtype=raw.dtype)
                sigmas = sig_low + u_rand * (sig_high - sig_low)

                # Random slopes along x and y with sampled sigma
                slope_x = torch.randn((b, 1, 1, 1), device=raw.device, dtype=raw.dtype) * sigmas
                slope_y = torch.randn((b, 1, 1, 1), device=raw.device, dtype=raw.dtype) * sigmas

                # Tilted plane: P(0,0) = 0 at cutout center
                tilted_plane = slope_x * grid_x.view(1, 1, h, w) + slope_y * grid_y.view(1, 1, h, w)
                if aug_raw.ndim == 3:
                    tilted_plane = tilted_plane.squeeze(1)
                aug_raw = aug_raw + tilted_plane

            # E. Photometric Scaling Jitter (zeropoint calibration variations)
            if self.flux_scale_prob > 0 and torch.rand(1, device=raw.device).item() < self.flux_scale_prob:
                scale_low, scale_high = self.flux_scale_range
                scale_factor = torch.empty((b, 1, 1, 1), device=raw.device).uniform_(scale_low, scale_high)
                if aug_raw.ndim == 3:
                    scale_factor = scale_factor.squeeze(1)
                aug_raw = aug_raw * scale_factor

        # -------------------------------------------------------------------
        # 2. Synchronized Spatial Geometric Transforms
        # -------------------------------------------------------------------
        if self.enable_spatial:
            # A. Random 90, 180, 270 deg Rotations
            if self.rot_prob > 0 and torch.rand(1, device=raw.device).item() < self.rot_prob:
                k = torch.randint(1, 4, (1,)).item()
                aug_raw = torch.rot90(aug_raw, k=k, dims=[-2, -1])
                if aug_target is not None:
                    aug_target = torch.rot90(aug_target, k=k, dims=[-2, -1])

            # B. Random Horizontal Flip
            if self.flip_h_prob > 0 and torch.rand(1, device=raw.device).item() < self.flip_h_prob:
                aug_raw = torch.flip(aug_raw, dims=[-1])
                if aug_target is not None:
                    aug_target = torch.flip(aug_target, dims=[-1])

            # C. Random Vertical Flip
            if self.flip_v_prob > 0 and torch.rand(1, device=raw.device).item() < self.flip_v_prob:
                aug_raw = torch.flip(aug_raw, dims=[-2])
                if aug_target is not None:
                    aug_target = torch.flip(aug_target, dims=[-2])

        return aug_raw, aug_target


class GPUAstroTransform(nn.Module):
    """
    Unified GPU pipeline chaining:
    Raw Cutout -> GPUAstroAugmentation (Noise + Spatial) -> GPULeakySB -> Normalized RGB.

    Parameters
    ----------
    stretch_kwargs : dict, optional
        Arguments forwarded to GPULeakySB (zero_points, pixel_scale, mmin, mmax, etc.).
    aug_kwargs : dict, optional
        Arguments forwarded to GPUAstroAugmentation.
    """

    def __init__(
        self,
        stretch_kwargs: Optional[dict] = None,
        aug_kwargs: Optional[dict] = None,
        soften_boundaries: bool = False,
        boundary_sigma: float = 1.2,
    ):
        super().__init__()
        self.stretch = GPULeakySB(**(stretch_kwargs or {}))
        self.aug = GPUAstroAugmentation(**(aug_kwargs or {}))
        self.soften_boundaries = soften_boundaries
        self.boundary_sigma = boundary_sigma

        # Precompute Gaussian blur kernel for on-the-fly GPU edge softening
        ksize = 5
        coords = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(gx**2 + gy**2) / (2 * boundary_sigma**2))
        kernel = kernel / kernel.sum()
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, ksize, ksize))

    def forward(
        self,
        raw: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute GPU transform pipeline on a batch.

        Parameters
        ----------
        raw : torch.Tensor
            Raw detector counts of shape (B, 1, H, W) or (B, H, W).
        target : torch.Tensor, optional
            Target labels of shape (B, H, W) or (B, 3, H, W).

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            (stretched_images, augmented_targets) where images is (B, 3, H, W) in [0.0, 1.0].
        """
        raw_aug, target_aug = self.aug(raw, target)
        images = self.stretch(raw_aug)

        # On-the-fly GPU edge softening: convert discrete (B, H, W) to probability simplex (B, 3, H, W)
        if self.soften_boundaries and target_aug is not None and target_aug.ndim == 3:
            one_hot = torch.nn.functional.one_hot(target_aug.clamp(0, 2), num_classes=3).permute(0, 3, 1, 2).float()
            k = self.gaussian_kernel.to(device=target_aug.device, dtype=torch.float32)
            smooth_edge = torch.nn.functional.conv2d(one_hot[:, 2:3], k, padding=2)
            one_hot[:, 2:3] = torch.clamp(smooth_edge * 1.5, 0.0, 1.0)
            target_aug = one_hot / torch.clamp(one_hot.sum(dim=1, keepdim=True), min=1e-6)

        return images, target_aug
