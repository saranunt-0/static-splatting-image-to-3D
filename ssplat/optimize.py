"""Single-view refinement: fit the visible-surface Gaussians to the photo.

What can (and cannot) be learned from one image: the photo constrains colour,
opacity and the screen-space footprint of every Gaussian, but nothing about
depth. Unconstrained 3DGS training on a single view therefore "overfits" by
sliding Gaussians along their viewing rays and stretching them into needles,
which looks perfect from the input camera and broken from anywhere else.

So the refinement keeps the geometry on the monocular estimate:
  * depth loss: rendered depth must match the (edge-sharpened) depth map,
  * anchor loss: each Gaussian keeps its distance along its viewing ray,
  * flatten loss: Gaussians stay thin, surface-like disks,
and only the visible layer is optimised (the hidden layer is not seen by this
camera, so the photo carries no information about it).

What it buys: a sharper, exact reproduction of the photo (the initial per-pixel
Gaussians blur it slightly) and soft, anti-aliased object boundaries learned as
partial opacity. docs/RESEARCH.md quantifies the effect on novel views.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np
import torch

from .config import OptimConfig
from .depth import _device
from .gaussians import GaussianSet
from .metrics import psnr, ssim_map
from .rasterize import Camera, rasterize


def _log(msg: str) -> None:
    print(f"[optimize] {msg}", flush=True)


def _logit(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def refine(gs: GaussianSet, image: np.ndarray, depth: np.ndarray, K: np.ndarray, cfg: OptimConfig,
           median_depth: float) -> Tuple[GaussianSet, Dict]:
    """image (H, W, 3) float in [0, 1]; depth (H, W) target depth; K intrinsics of `image`."""
    dev = _device(cfg.device)
    torch.manual_seed(cfg.seed)
    H, W = depth.shape
    t = gs.to_torch(dev)
    train = t["layer"] == 0
    fixed = {k: v[~train] for k, v in t.items()}

    means0 = t["means"][train].clone()
    z0 = means0.norm(dim=1, keepdim=True).clamp_min(1e-6)
    ray = means0 / z0
    params = {
        "means": means0.clone().requires_grad_(True),
        "log_scales": torch.log(t["scales"][train]).requires_grad_(True),
        "quats": t["quats"][train].clone().requires_grad_(True),
        "op_logit": _logit(t["opacities"][train]).requires_grad_(True),
        "colors": t["colors"][train].clone().requires_grad_(True),
    }
    lrs = {"means": cfg.lr_means * median_depth, "log_scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "op_logit": cfg.lr_opacity, "colors": cfg.lr_color}
    opt = torch.optim.Adam([{"params": [p], "lr": lrs[k], "name": k} for k, p in params.items()], eps=1e-15)

    s = cfg.render_scale
    cam = Camera(torch.tensor(K, dtype=torch.float32), torch.eye(4), W, H).to(dev)
    cam = cam.scaled(s) if s != 1.0 else cam
    img = torch.as_tensor(image, dtype=torch.float32, device=dev)
    dep = torch.as_tensor(depth, dtype=torch.float32, device=dev)
    if s != 1.0:
        img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(cam.height, cam.width),
                                              mode="area")[0].permute(1, 2, 0)
        dep = torch.nn.functional.interpolate(dep[None, None], size=(cam.height, cam.width), mode="area")[0, 0]

    def render(p):
        m = torch.cat([p["means"], fixed["means"]])
        q = torch.cat([p["quats"], fixed["quats"]])
        sc = torch.cat([torch.exp(p["log_scales"]), fixed["scales"]])
        o = torch.cat([torch.sigmoid(p["op_logit"]), fixed["opacities"]])
        c = torch.cat([p["colors"], fixed["colors"]]).clamp_min(0.0)  # viewers clamp at 0 too
        return rasterize(m, q, sc, o, c, cam)

    with torch.no_grad():
        out0 = render(params)
        psnr0 = psnr(out0["rgb"], img)
    stats = {"psnr_input_view_before": round(psnr0, 2),
             "l1_input_view_before": round(float((out0["rgb"] - img).abs().mean()), 5),
             "steps": cfg.steps, "device": str(dev)}
    _log(f"{int(train.sum()):,} trainable + {int((~train).sum()):,} fixed Gaussians on {dev}; "
         f"input-view PSNR before: {psnr0:.2f} dB")
    t_start = time.time()
    for step in range(cfg.steps):
        frac = step / max(cfg.steps, 1)
        for g in opt.param_groups:  # cosine decay to 10%
            g["lr"] = lrs[g["name"]] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        out = render(params)
        rgb, alpha = out["rgb"], out["alpha"]
        l1 = (rgb - img).abs().mean()
        loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - ssim_map(rgb, img).mean())
        if cfg.depth_weight > 0:
            d = out["depth"] / alpha.clamp_min(1e-6)
            m = alpha > 0.5
            loss = loss + cfg.depth_weight * ((d - dep).abs() / dep)[m].mean()
        if cfg.anchor_weight > 0:
            along = ((params["means"] - means0) * ray).sum(1, keepdim=True) / z0
            loss = loss + cfg.anchor_weight * (along ** 2).mean() * 100.0
        if cfg.flatten_weight > 0:
            sc = torch.exp(params["log_scales"])
            loss = loss + cfg.flatten_weight * (sc.min(1).values / sc.max(1).values).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == cfg.steps - 1:
            el = time.time() - t_start
            _log(f"step {step:4d}/{cfg.steps}  loss {loss.item():.4f}  L1 {l1.item():.4f}  "
                 f"{el / (step + 1):.2f}s/step")

    with torch.no_grad():
        out1 = render(params)
        stats["psnr_input_view_after"] = round(psnr(out1["rgb"], img), 2)
        stats["l1_input_view_after"] = round(float((out1["rgb"] - img).abs().mean()), 5)
        stats["seconds"] = round(time.time() - t_start, 1)
        along = ((params["means"] - means0) * ray).sum(1) / z0[:, 0]
        stats["median_rel_depth_shift"] = round(float(along.abs().median()), 5)
        new = {
            "means": params["means"], "quats": params["quats"] / params["quats"].norm(dim=1, keepdim=True),
            "scales": torch.exp(params["log_scales"]), "opacities": torch.sigmoid(params["op_logit"]),
            "colors": params["colors"].clamp_min(0.0), "layer": t["layer"][train],
        }
        merged = {k: torch.cat([new[k], fixed[k]]) for k in new}
    _log(f"input-view PSNR {stats['psnr_input_view_before']} -> {stats['psnr_input_view_after']} dB "
         f"in {stats['seconds']} s")
    return GaussianSet.from_torch(merged), stats
