"""Depth map -> Gaussians: one surface-aligned Gaussian per pixel."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import LiftConfig
from .gaussians import GaussianSet
from .geometry import Edges, footprint_gaussians, occlusion_edges, unproject


def lift(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, cut_x: np.ndarray, cut_y: np.ndarray,
         cfg: LiftConfig, mask: Optional[np.ndarray] = None, layer: int = 0,
         offset=(0.0, 0.0)) -> GaussianSet:
    """rgb (H, W, 3) in [0, 1], depth (H, W); only pixels in `mask` (default: all) become Gaussians."""
    P = unproject(depth, K, offset)
    scales, quats = footprint_gaussians(P, K, cut_x, cut_y, cfg.footprint_sigma, cfg.thickness)
    sel = np.ones(depth.shape, bool) if mask is None else mask
    flat = sel.reshape(-1)
    n = int(flat.sum())
    return GaussianSet(
        means=P.reshape(-1, 3)[flat].astype(np.float32),
        quats=quats[flat],
        scales=scales[flat],
        opacities=np.full(n, cfg.opacity, np.float32),
        colors=rgb.reshape(-1, 3)[flat].astype(np.float32),
        layer=np.full(n, layer, np.int8),
    )


def lift_surface(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, edges: Edges, cfg: LiftConfig) -> GaussianSet:
    return lift(rgb, depth, K, edges.cut_x, edges.cut_y, cfg)


def lift_layer(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, mask: np.ndarray, cfg: LiftConfig,
               baseline: float, parallax_px: float, max_angle_deg: float, layer: int = 1) -> GaussianSet:
    """Lift a partial layer (pixels in `mask`); neighbours outside the mask are never joined."""
    d = np.where(mask, depth, np.nanmax(np.where(mask, depth, np.nan)) if mask.any() else 1.0)
    e = occlusion_edges(d, K[0, 0], baseline, parallax_px, max_angle_deg)
    cut_x = e.cut_x | ~(mask[:, :-1] & mask[:, 1:])
    cut_y = e.cut_y | ~(mask[:-1, :] & mask[1:, :])
    return lift(rgb, d, K, cut_x, cut_y, cfg, mask=mask, layer=layer)
