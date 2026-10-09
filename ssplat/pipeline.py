"""Core reconstruction (no file I/O): image + depth -> GaussianSet.

Used by the CLI (ssplat/run.py), the notebook and the benchmark (bench/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .config import PipelineConfig
from .gaussians import GaussianSet
from .geometry import occlusion_edges, sharpen_depth_edges, unproject
from .lift import lift_layer, lift_surface


@dataclass
class Reconstruction:
    gaussians: GaussianSet
    K: np.ndarray
    width: int
    height: int
    median_depth: float
    depth: np.ndarray                 # depth map actually used (after edge sharpening)
    hidden: Optional[object] = None   # occlusion.HiddenLayer
    info: Dict = field(default_factory=dict)


def naive_points(img_f: np.ndarray, depth: np.ndarray, K: np.ndarray, sigma_px: float = 0.6,
                 opacity: float = 0.98) -> GaussianSet:
    """Baseline: back-projected point cloud as isotropic Gaussians sized to the pixel footprint."""
    P = unproject(depth, K).reshape(-1, 3).astype(np.float32)
    s = (sigma_px * P[:, 2] / K[0, 0]).astype(np.float32)
    n = P.shape[0]
    q = np.zeros((n, 4), np.float32)
    q[:, 0] = 1
    return GaussianSet(P, q, np.repeat(s[:, None], 3, 1), np.full(n, opacity, np.float32),
                       img_f.reshape(-1, 3).astype(np.float32))


def reconstruct(img: np.ndarray, depth: np.ndarray, K: np.ndarray, cfg: PipelineConfig,
                sky: Optional[np.ndarray] = None, inpainter=None) -> Reconstruction:
    """img (H, W, 3) uint8, depth (H, W) z-depth, K intrinsics."""
    from .occlusion import build_hidden_layer
    from .optimize import refine

    H, W = depth.shape
    sky = np.zeros((H, W), bool) if sky is None else sky
    median = float(np.median(depth[~sky])) if (~sky).any() else float(np.median(depth))
    oc = cfg.occlusion
    b = oc.max_baseline * median
    if cfg.lift.sharpen_edges:
        depth = sharpen_depth_edges(depth, K[0, 0], b, oc.edge_parallax_px, oc.edge_min_ratio)
    edges = occlusion_edges(depth, K[0, 0], b, oc.edge_parallax_px, oc.edge_min_ratio)
    img_f = img.astype(np.float32) / 255.0
    sets = [lift_surface(img_f, depth, K, edges, cfg.lift)]
    hidden = None
    info: Dict = {"median_depth": median}
    if oc.enabled:
        if inpainter is None:
            from .inpaint import make_inpainter

            inpainter = make_inpainter(oc.inpainter, oc.lama_checkpoint, cfg.optim.device)
        hidden = build_hidden_layer(img, depth, K, sky, oc, inpainter, median)
        sets.append(lift_layer(hidden.rgb.astype(np.float32) / 255.0, hidden.depth, hidden.K, hidden.mask,
                               cfg.lift, b, oc.edge_parallax_px, oc.edge_min_ratio, layer=1))
        info["hidden_layer"] = hidden.stats()
    gs = GaussianSet.concat(*sets)
    if cfg.optim.steps > 0:
        gs, info["optim"] = refine(gs, img_f, depth, K, cfg.optim, median)
    info["num_gaussians"] = {"visible": int((gs.layer == 0).sum()), "hidden": int((gs.layer == 1).sum())}
    return Reconstruction(gs, K, W, H, median, depth, hidden, info)
