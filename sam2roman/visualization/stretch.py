
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tv_F

ArrayOrTensor = np.ndarray | torch.Tensor


def _norm_leaky_sb_forward_1d(
    f_norm: torch.Tensor,
    span: float,
    negative_slope: float,
    positive_slope: float,
    y_zero: float = 0.015,
    y_one: float = 0.96,
) -> torch.Tensor:
    """
    Transformación 1D biyectiva: mapea todo flujo (-inf, +inf) dentro de (0.0, 1.0).
    `y_zero = 0.015` mantiene el fondo del cielo negro profundo.
    """
    mid_span = y_one - y_zero
    k_tangent = (2.5 / np.log(10.0)) / span
    top_room = 1.0 - y_one

    f_safe = torch.clamp(f_norm, min=1.0)
    u = (2.5 * torch.log10(f_safe)) / span

    # Régimen logarítmico normal [mmax, mmin] -> [y_zero, y_one]
    y_mid = y_zero + mid_span * u

    # Régimen brillante (< mmin) -> (y_one, 1.0)
    s_pos = (positive_slope * mid_span) / max(top_room, 1e-8)
    u_excess = torch.clamp(u - 1.0, min=0.0)
    y_high = 1.0 - (top_room / (1.0 + s_pos * u_excess))
    y_pos_branch = torch.where(u <= 1.0, y_mid, y_high)

    # Régimen débil y negativo (> mmax y flujos < 0) -> (0.0, y_zero)
    s_neg = (negative_slope * k_tangent * mid_span) / max(y_zero, 1e-8)
    f_deficit = torch.clamp(1.0 - f_norm, min=0.0)
    y_low = y_zero / (1.0 + s_neg * f_deficit)

    return torch.where(f_norm >= 1.0, y_pos_branch, y_low)


def _norm_leaky_sb_inverse_1d(
    y: torch.Tensor,
    span: float,
    negative_slope: float,
    positive_slope: float,
    y_zero: float = 0.015,
    y_one: float = 0.96,
) -> torch.Tensor:
    """Inversa analítica exacta de `_norm_leaky_sb_forward_1d`."""
    mid_span = y_one - y_zero
    k_tangent = (2.5 / np.log(10.0)) / span
    top_room = 1.0 - y_one

    s_pos = (positive_slope * mid_span) / max(top_room, 1e-8)
    s_neg = (negative_slope * k_tangent * mid_span) / max(y_zero, 1e-8)

    u_mid = (y - y_zero) / mid_span
    y_high_safe = torch.clamp(1.0 - y, min=1e-14)
    u_high = 1.0 + ((top_room / y_high_safe) - 1.0) / s_pos

    u = torch.where(y <= y_one, u_mid, u_high)
    f_pos_branch = 10.0 ** ((u * span) / 2.5)

    y_low_safe = torch.clamp(y, min=1e-14)
    f_low = 1.0 - (((y_zero / y_low_safe) - 1.0) / s_neg)

    return torch.where(y >= y_zero, f_pos_branch, f_low)


def _soft_box_forward(x: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    """
    Comprime suavemente cualquier exceso de crominancia fuera de [margin, 1 - margin]
    hacia (0.0, 1.0) de forma 100% biyectiva e invertible.
    """
    low_lim = margin
    high_lim = 1.0 - margin

    # Compresión suave por debajo de low_lim hacia 0.0
    x_low = low_lim / (1.0 + (low_lim - torch.clamp(x, max=low_lim)) / low_lim)
    # Compresión suave por encima de high_lim hacia 1.0
    x_high = 1.0 - margin / (1.0 + (torch.clamp(x, min=high_lim) - high_lim) / margin)

    out = torch.where(x < low_lim, x_low, x)
    out = torch.where(x > high_lim, x_high, out)
    return out


def _soft_box_inverse(y: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    """Inversa exacta de `_soft_box_forward`."""
    low_lim = margin
    high_lim = 1.0 - margin

    y_low_safe = torch.clamp(y, min=1e-14)
    x_low = low_lim - low_lim * ((low_lim / y_low_safe) - 1.0)

    y_high_safe = torch.clamp(1.0 - y, min=1e-14)
    x_high = high_lim + margin * ((margin / y_high_safe) - 1.0)

    out = torch.where(y < low_lim, x_low, y)
    out = torch.where(y > high_lim, x_high, out)
    return out


def _chroma_weight(
    lum_n: torch.Tensor,
    y_lum: torch.Tensor,
    span: float,
    negative_slope: float,
    y_zero: float,
    color_boost: float,
) -> torch.Tensor:
    """
    Peso de crominancia estrictamente positivo `w > 0` que preserva las proporciones
    reales de color (estilo Lupton) como en la imagen de la derecha.
    """
    k_tangent = (2.5 / np.log(10.0)) / span
    slope_neg = negative_slope * k_tangent
    w_base = torch.where(
        lum_n >= 1.0,
        ((y_lum - y_zero) + slope_neg) / torch.clamp(lum_n, min=1.0),
        torch.full_like(lum_n, slope_neg),
    )
    return w_base * color_boost


def leakySB(
    r: ArrayOrTensor,
    g: ArrayOrTensor,
    b: ArrayOrTensor,
    zero_points: float | Sequence[float] = 31.4,
    pixel_scale: float | Sequence[float] = 0.262,
    mmin: float = 13.0,
    mmax: float = 32.0,
    negative_slope: float = 0.1,
    positive_slope: float = 0.05,
    color_boost: float = 1.2,
    preserve_color_ratios: bool = True,
    y_zero: float = 0.015,
    y_one: float = 0.96,
    channels_last: bool = True,
    output_dtype: torch.dtype | None = torch.float64,
) -> ArrayOrTensor:
    """
    Convierte flujos FITS R, G, B en una imagen RGB en mag/arcsec^2 normalizada
    estrictamente en (0.0, 1.0), preservando el alto contraste de color (como en la
    imagen de la derecha) y siendo 100% reversible.
    """
    is_numpy = isinstance(r, np.ndarray)
    device = r.device if isinstance(r, torch.Tensor) else "cpu"
    span = float(mmax - mmin)

    zps = (zero_points,) * 3 if isinstance(zero_points, (int, float)) else tuple(zero_points)
    pss = (pixel_scale,) * 3 if isinstance(pixel_scale, (int, float)) else tuple(pixel_scale)

    norm_bands = []
    for ch, zp, ps in zip((r, g, b), zps, pss):
        ch_arr = np.asarray(ch, dtype=np.float64) if isinstance(ch, np.ndarray) else ch
        t = torch.as_tensor(ch_arr, dtype=torch.float64, device=device)
        if torch.isnan(t).any() or torch.isinf(t).any():
            finite_mask = torch.isfinite(t)
            fill_val = torch.median(t[finite_mask]) if finite_mask.any() else torch.tensor(0.0, dtype=torch.float64, device=device)
            t = torch.where(finite_mask, t, fill_val)
        f_mmax = (float(ps) ** 2) * (10.0 ** (0.4 * (float(zp) - float(mmax))))
        norm_bands.append(t / f_mmax)

    r_n, g_n, b_n = norm_bands
    stack_dim = -1 if channels_last else 0

    if preserve_color_ratios:
        lum_n = (r_n + g_n + b_n) / 3.0
        y_lum = _norm_leaky_sb_forward_1d(
            lum_n, span, negative_slope, positive_slope, y_zero=y_zero, y_one=y_one
        )
        w = _chroma_weight(lum_n, y_lum, span, negative_slope, y_zero, color_boost)

        # Luminancia + Crominancia preservando proporciones de flujo R:G:B
        r_raw = y_lum + w * (r_n - lum_n)
        g_raw = y_lum + w * (g_n - lum_n)
        b_raw = y_lum + w * (b_n - lum_n)

        # Garantizar que los 3 canales estén estrictamente dentro de (0.0, 1.0) sin perder invertibilidad
        rgb = torch.stack(
            [
                _soft_box_forward(r_raw),
                _soft_box_forward(g_raw),
                _soft_box_forward(b_raw),
            ],
            dim=stack_dim,
        )
    else:
        rgb = torch.stack(
            [
                _norm_leaky_sb_forward_1d(r_n, span, negative_slope, positive_slope, y_zero, y_one),
                _norm_leaky_sb_forward_1d(g_n, span, negative_slope, positive_slope, y_zero, y_one),
                _norm_leaky_sb_forward_1d(b_n, span, negative_slope, positive_slope, y_zero, y_one),
            ],
            dim=stack_dim,
        )

    if output_dtype is not None:
        rgb = rgb.to(output_dtype)

    return rgb.detach().cpu().numpy() if is_numpy else rgb


def leakySB_inverse(
    rgb: ArrayOrTensor,
    zero_points: float | Sequence[float] = 31.4,
    pixel_scale: float | Sequence[float] = 0.262,
    mmin: float = 13.0,
    mmax: float = 32.0,
    negative_slope: float = 0.1,
    positive_slope: float = 0.05,
    color_boost: float = 1.2,
    preserve_color_ratios: bool = True,
    y_zero: float = 0.015,
    y_one: float = 0.96,
) -> tuple[ArrayOrTensor, ArrayOrTensor, ArrayOrTensor]:
    """
    Inversa exacta de `leakySB`. Recupera los flujos FITS originales con error ~0.
    """
    is_numpy = isinstance(rgb, np.ndarray)
    rgb_arr = np.asarray(rgb, dtype=np.float64) if isinstance(rgb, np.ndarray) else rgb
    t = torch.as_tensor(rgb_arr, dtype=torch.float64)
    span = float(mmax - mmin)

    if t.ndim == 3 and t.shape[-1] == 3:
        r_box, g_box, b_box = t[..., 0], t[..., 1], t[..., 2]
    elif t.ndim == 3 and t.shape[0] == 3:
        r_box, g_box, b_box = t[0], t[1], t[2]
    else:
        raise ValueError(f"Expected 3D RGB image, got {t.shape}")

    zps = (zero_points,) * 3 if isinstance(zero_points, (int, float)) else tuple(zero_points)
    pss = (pixel_scale,) * 3 if isinstance(pixel_scale, (int, float)) else tuple(pixel_scale)

    if preserve_color_ratios:
        # 1. Deshacer la compresión suave a (0.0, 1.0)
        r_raw = _soft_box_inverse(r_box)
        g_raw = _soft_box_inverse(g_box)
        b_raw = _soft_box_inverse(b_box)

        # 2. La media exacta de (r_raw, g_raw, b_raw) es idéntica a `y_lum`
        y_lum = (r_raw + g_raw + b_raw) / 3.0

        # 3. Invertir la luminancia 1D para obtener `lum_n`
        lum_n = _norm_leaky_sb_inverse_1d(
            y_lum, span, negative_slope, positive_slope, y_zero=y_zero, y_one=y_one
        )

        # 4. Recalcular el peso de crominancia `w` y despejar cada canal
        w = _chroma_weight(lum_n, y_lum, span, negative_slope, y_zero, color_boost)
        r_n = lum_n + (r_raw - y_lum) / w
        g_n = lum_n + (g_raw - y_lum) / w
        b_n = lum_n + (b_raw - y_lum) / w
        norm_bands = (r_n, g_n, b_n)
    else:
        norm_bands = tuple(
            _norm_leaky_sb_inverse_1d(ch, span, negative_slope, positive_slope, y_zero, y_one)
            for ch in (r_box, g_box, b_box)
        )

    orig_bands = []
    for ch_n, zp, ps in zip(norm_bands, zps, pss):
        f_mmax = (float(ps) ** 2) * (10.0 ** (0.4 * (float(zp) - float(mmax))))
        orig_flux = (ch_n * f_mmax).to(torch.float32)
        orig_bands.append(orig_flux.detach().cpu().numpy() if is_numpy else orig_flux)

    return orig_bands[0], orig_bands[1], orig_bands[2]


def make_cmap(max_label, background_color="#000000ff", seed=None):

    from matplotlib import colors

    cmap = make_random_cmap(max_label + 1, seed=seed)

    if background_color is not None:
        cmap.colors[0] = colors.to_rgba(background_color)

    return cmap


def make_random_cmap(ncolors=256, seed=None):
    """
    Make a matplotlib colormap consisting of (random) muted colors.

    A random colormap is very useful for plotting segmentation images.

    Parameters
    ----------
    ncolors : int, optional
        The number of colors in the colormap.  The default is 256.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.
        Separate function calls with the same ``seed`` will generate the
        same colormap.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
        The matplotlib colormap with random colors in RGBA format.
    """
    from matplotlib import colors

    rng = np.random.default_rng(seed)
    hue = rng.uniform(low=0.0, high=1.0, size=ncolors)
    sat = rng.uniform(low=0.2, high=0.7, size=ncolors)
    val = rng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((hue, sat, val))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))

    return colors.ListedColormap(colors.to_rgba_array(rgb))
