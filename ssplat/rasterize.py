"""Differentiable 3D Gaussian Splatting rasterizer in pure PyTorch.

Why not gsplat / the Inria CUDA rasterizer? This project reconstructs a single
image, so the scene is small (one Gaussian per pixel) and a few hundred
optimisation steps are enough. A pure-PyTorch renderer then runs fast enough on
a Colab GPU *and* still works on a CPU-only machine, with no CUDA compilation
step and identical results everywhere.

The image formation model is exactly the standard 3DGS one (EWA splatting with
the 0.3 px^2 screen-space dilation, alpha = min(0.99, o * G), Gaussians with
alpha < 1/255 skipped, front-to-back compositing, early stop at T < 1e-4), so
the exported .ply files look the same in every standard 3DGS viewer.

Implementation: instead of tiles, every Gaussian is expanded into the pixels of
its own 3-sigma window ("fragments"). Fragments are sorted by (pixel, depth)
and composited with a segmented cumulative sum of log(1 - alpha), which is
exact, fully vectorised and differentiable with plain autograd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_MIN = 1e-4
DILATION = 0.3  # screen-space low-pass filter of the reference 3DGS renderer (px^2)
# Window half-sizes (px). Each Gaussian uses the smallest window covering its 3-sigma radius.
_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


@dataclass
class Camera:
    """Pinhole camera, OpenCV convention (x right, y down, z forward)."""

    K: torch.Tensor        # (3, 3) intrinsics in pixels; pixel centres at integer + 0.5
    viewmat: torch.Tensor  # (4, 4) world -> camera
    width: int
    height: int

    def to(self, device) -> "Camera":
        return Camera(self.K.to(device), self.viewmat.to(device), self.width, self.height)

    def scaled(self, factor: float) -> "Camera":
        """Same camera at a different image resolution."""
        K = self.K.clone()
        K[:2] *= factor
        return Camera(K, self.viewmat, int(round(self.width * factor)), int(round(self.height * factor)))


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) quaternions (w, x, y, z), any norm -> (N, 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def project(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor, cam: Camera,
            near: float = 1e-3) -> Dict[str, torch.Tensor]:
    """EWA projection of 3D Gaussians: screen-space means, conics, radii, depths."""
    R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
    pc = means @ R.T + t
    z = pc[:, 2]
    zc = z.clamp_min(near)
    fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
    # Clamp the Jacobian's evaluation point to a slightly enlarged frustum (as in 3DGS),
    # otherwise Gaussians far outside the image get huge, unstable footprints.
    tx = zc * torch.clamp(pc[:, 0] / zc, -1.3 * cx / fx, 1.3 * (cam.width - cx) / fx)
    ty = zc * torch.clamp(pc[:, 1] / zc, -1.3 * cy / fy, 1.3 * (cam.height - cy) / fy)
    zero = torch.zeros_like(zc)
    J = torch.stack([
        fx / zc, zero, -fx * tx / (zc * zc),
        zero, fy / zc, -fy * ty / (zc * zc),
    ], dim=-1).reshape(-1, 2, 3)
    M = quat_to_rotmat(quats) * scales[:, None, :]          # (N,3,3) = R_q diag(s)
    T = J @ R[None] @ M                                       # (N,2,3)
    cov = T @ T.transpose(1, 2)                               # (N,2,2) screen covariance
    a = cov[:, 0, 0] + DILATION
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + DILATION
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.01))
    radius = torch.ceil(3.0 * torch.sqrt(lam))
    mean2d = torch.stack([fx * pc[:, 0] / zc + cx, fy * pc[:, 1] / zc + cy], dim=-1)
    return {"mean2d": mean2d, "conic": conic, "radius": radius.detach(), "depth": z}


@torch.no_grad()
def _fragments(mean2d, conic, opacity, depth, radius, width, height, near, max_radius, chunk=2_000_000):
    """Indices (gaussian, pixel) of all fragments with alpha >= 1/255, sorted by (pixel, depth)."""
    device = mean2d.device
    r = radius.clamp(max=max_radius)
    visible = (depth > near) & (r > 0)
    visible &= (mean2d[:, 0] + r > 0) & (mean2d[:, 0] - r < width)
    visible &= (mean2d[:, 1] + r > 0) & (mean2d[:, 1] - r < height)
    # Depth rank used as the secondary sort key.
    vis_idx = visible.nonzero().squeeze(1)
    order = torch.argsort(depth[vis_idx])
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=device)
    rank_full = torch.zeros(mean2d.shape[0], dtype=torch.long, device=device)
    rank_full[vis_idx] = rank
    n_rank = max(int(order.numel()), 1)

    g_all, p_all = [], []
    lo = 0
    for h in _BUCKETS:
        sel = vis_idx[(r[vis_idx] > lo) & (r[vis_idx] <= h)]
        lo = h
        if sel.numel() == 0:
            continue
        side = 2 * h + 1
        off = torch.arange(-h, h + 1, device=device)
        oy, ox = torch.meshgrid(off, off, indexing="ij")
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        per = max(1, chunk // (side * side))
        for s in range(0, sel.numel(), per):
            g = sel[s:s + per]
            px = torch.floor(mean2d[g, 0])[:, None].long() + ox[None]
            py = torch.floor(mean2d[g, 1])[:, None].long() + oy[None]
            dx = px.float() + 0.5 - mean2d[g, 0:1]
            dy = py.float() + 0.5 - mean2d[g, 1:2]
            co = conic[g]
            power = -0.5 * (co[:, 0:1] * dx * dx + co[:, 2:3] * dy * dy) - co[:, 1:2] * dx * dy
            alpha = opacity[g, None] * torch.exp(power)
            rg = r[g, None]
            ok = (alpha >= ALPHA_MIN) & (power <= 0) & (dx.abs() <= rg) & (dy.abs() <= rg)
            ok &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
            gg = g[:, None].expand_as(px)[ok]
            g_all.append(gg)
            p_all.append((py * width + px)[ok])
        if h >= max_radius:
            break
    if not g_all:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty
    g = torch.cat(g_all)
    p = torch.cat(p_all)
    key = p * n_rank + rank_full[g]
    perm = torch.argsort(key)
    return g[perm], p[perm]


def rasterize(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor, opacities: torch.Tensor,
              colors: torch.Tensor, cam: Camera, background: Optional[torch.Tensor] = None,
              near: float = 1e-3, max_radius: int = 128, extra: Optional[torch.Tensor] = None
              ) -> Dict[str, torch.Tensor]:
    """Render Gaussians.

    Args:
        means (N,3), quats (N,4) wxyz, scales (N,3) (activated, i.e. sigma), opacities (N,) in [0,1],
        colors (N,C) linear values in [0,1] (C=3 usually), cam: Camera.
        extra: optional (N,E) per-Gaussian features rendered with the same weights.
    Returns:
        dict with "rgb" (H,W,C), "depth" (H,W) expected z-depth (un-normalised, i.e.
        sum w z; divide by alpha for the surface depth), "alpha" (H,W),
        "extra" (H,W,E) if requested, and "n_fragments".
    """
    H, W = cam.height, cam.width
    proj = project(means, quats, scales, cam, near)
    g, p = _fragments(proj["mean2d"], proj["conic"], opacities, proj["depth"], proj["radius"],
                      W, H, near, max_radius)
    C = colors.shape[1]
    device, dtype = means.device, means.dtype
    rgb = torch.zeros(H * W, C, device=device, dtype=dtype)
    dep = torch.zeros(H * W, device=device, dtype=dtype)
    acc = torch.zeros(H * W, device=device, dtype=dtype)
    ext = None if extra is None else torch.zeros(H * W, extra.shape[1], device=device, dtype=dtype)
    if g.numel() > 0:
        px = (p % W).to(dtype) + 0.5
        py = torch.div(p, W, rounding_mode="floor").to(dtype) + 0.5
        m = proj["mean2d"][g]
        co = proj["conic"][g]
        dx, dy = px - m[:, 0], py - m[:, 1]
        power = (-0.5 * (co[:, 0] * dx * dx + co[:, 2] * dy * dy) - co[:, 1] * dx * dy).clamp_max(0.0)
        alpha = (opacities[g] * torch.exp(power)).clamp(max=ALPHA_MAX)
        # Segmented exclusive cumulative sum of log(1 - alpha) per pixel (float64 for accuracy).
        log1m = torch.log1p(-alpha).double()
        incl = torch.cumsum(log1m, 0)
        excl = incl - log1m
        _, counts = torch.unique_consecutive(p, return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        base = torch.repeat_interleave(excl[starts], counts)
        T = torch.exp(excl - base).to(dtype)
        # Early termination as in the reference renderer: a fragment is skipped (and so is
        # everything behind it) once it would push the transmittance below T_MIN.
        keep = (incl - base) >= torch.log(torch.tensor(T_MIN, dtype=torch.float64))
        w = alpha * T * keep.to(dtype)
        rgb = rgb.index_add(0, p, w[:, None] * colors[g])
        dep = dep.index_add(0, p, w * proj["depth"][g])
        acc = acc.index_add(0, p, w)
        if extra is not None:
            ext = ext.index_add(0, p, w[:, None] * extra[g])
    if background is not None:
        rgb = rgb + (1.0 - acc)[:, None] * background.to(dtype).reshape(1, C)
    out = {"rgb": rgb.reshape(H, W, C), "depth": dep.reshape(H, W), "alpha": acc.reshape(H, W),
           "n_fragments": torch.tensor(g.numel())}
    if ext is not None:
        out["extra"] = ext.reshape(H, W, -1)
    return out
