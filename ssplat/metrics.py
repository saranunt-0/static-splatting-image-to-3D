"""Image metrics (PSNR, SSIM) shared by the optimiser and the benchmark."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(H, W, C) images in [0, 1] -> (H, W) SSIM map (Wang et al. 2004, 11x11 Gaussian)."""
    x = a.permute(2, 0, 1)[:, None]
    y = b.permute(2, 0, 1)[:, None]
    w = _gauss_window(device=a.device, dtype=a.dtype)
    pad = w.shape[-1] // 2
    f = lambda t: F.conv2d(F.pad(t, (pad,) * 4, mode="replicate"), w)
    mx, my = f(x), f(y)
    sxx = f(x * x) - mx * mx
    syy = f(y * y) - my * my
    sxy = f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s[:, 0].mean(0)


def ssim(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    s = ssim_map(a, b)
    return float(s[mask].mean() if mask is not None else s.mean())


def psnr(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    d = (a.clamp(0, 1) - b.clamp(0, 1)) ** 2
    mse = d[mask].mean() if mask is not None else d.mean()
    return float(-10 * torch.log10(mse.clamp_min(1e-12)))
