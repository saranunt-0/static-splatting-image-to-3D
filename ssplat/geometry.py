"""Depth-map geometry: back-projection, occlusion edges, pixel-footprint Gaussians.

Conventions: pixel (row j, column i) has its centre at (i + 0.5, j + 0.5);
camera frame is OpenCV (x right, y down, z forward); `disp` = 1 / depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage


def unproject(depth: np.ndarray, K: np.ndarray, offset=(0.0, 0.0)) -> np.ndarray:
    """(H, W) z-depth -> (H, W, 3) camera-frame points. `offset` shifts pixel coordinates."""
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H) + 0.5 + offset[1], np.arange(W) + 0.5 + offset[0], indexing="ij")
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def sharpen_depth_edges(depth: np.ndarray, f: float, baseline: float, parallax_px: float,
                        max_angle_deg: float, iters: int = 3) -> np.ndarray:
    """Remove "flying pixels": snap depths in discontinuity zones to the nearer side.

    Monocular depth networks blur depth edges over a few pixels; lifted to 3D
    those pixels float in mid-air between foreground and background and show up
    as streaks as soon as the camera moves. A pixel is in a discontinuity zone if
    its 3x3 neighbourhood spans a jump that counts as an occlusion edge; its
    disparity is then replaced by the closer of the local minimum or maximum.
    """
    disp = 1.0 / depth
    # A 3x3 window spans up to ~3 pixel steps of a continuous surface, hence 3 * tan.
    steep = 3.0 * np.tan(np.radians(max_angle_deg))
    for _ in range(iters):
        lo = ndimage.minimum_filter(disp, size=3, mode="nearest")
        hi = ndimage.maximum_filter(disp, size=3, mode="nearest")
        zone = (f * baseline * (hi - lo) >= parallax_px) & (f * (hi - lo) >= steep * lo)
        # Near/far decision on the locally averaged disparity: spatially coherent, unlike a
        # per-pixel decision, which gives a dithered, salt-and-pepper edge.
        near = ndimage.uniform_filter(disp, size=3, mode="nearest") > 0.5 * (lo + hi)
        disp = np.where(zone, np.where(near, hi, lo), disp)
    return (1.0 / disp).astype(depth.dtype)


@dataclass
class Edges:
    """Occlusion edges between 4-neighbours.

    cut_x[j, i]: pixels (j, i) and (j, i+1) are on different surfaces; cut_y likewise
    for (j, i) and (j+1, i). fg / bg mark the near / far pixel of every cut pair and
    `gap` (at fg pixels) is the largest gap in pixels that opens behind it when the
    camera moves by `baseline`.
    """

    cut_x: np.ndarray
    cut_y: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    gap: np.ndarray


def occlusion_edges(depth: np.ndarray, f: float, baseline: float, parallax_px: float,
                    max_angle_deg: float = 88.0) -> Edges:
    """Depth discontinuities that would open a visible gap for camera motion <= baseline.

    Two physical tests, both required for a cut between neighbouring pixels:
    * visibility: moving the camera sideways by b shifts two points at disparities
      d1 > d2 apart by f * b * (d1 - d2) pixels; below `parallax_px` nobody can see it
      (e.g. a corridor floor near the vanishing point: tiny per-pixel steps);
    * steepness: a continuous surface seen at angle theta from its normal has a
      relative depth step of tan(theta) / f per pixel, i.e. f * (d1 - d2) / d2 = tan(theta).
      Anything steeper than `max_angle_deg` is treated as a jump, not a surface. This keeps
      near, steep but continuous surfaces (a shelf top seen almost edge-on) closed, and is
      independent of the image resolution.
    """
    disp = 1.0 / depth
    H, W = depth.shape
    fg = np.zeros((H, W), bool)
    bg = np.zeros((H, W), bool)
    gap = np.zeros((H, W), np.float32)

    steep = np.tan(np.radians(max_angle_deg))

    def pair(a, b):
        hi, lo = np.maximum(a, b), np.minimum(a, b)
        par = f * baseline * (hi - lo)
        return (par >= parallax_px) & (f * (hi - lo) >= steep * lo), par

    cut_x, par_x = pair(disp[:, :-1], disp[:, 1:])
    cut_y, par_y = pair(disp[:-1, :], disp[1:, :])
    left_near = disp[:, :-1] > disp[:, 1:]
    top_near = disp[:-1, :] > disp[1:, :]
    for cut, par, near_first, sl_a, sl_b in (
            (cut_x, par_x, left_near, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
            (cut_y, par_y, top_near, (slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        a_fg = cut & near_first
        b_fg = cut & ~near_first
        fg[sl_a] |= a_fg
        fg[sl_b] |= b_fg
        bg[sl_a] |= b_fg
        bg[sl_b] |= a_fg
        gap[sl_a] = np.maximum(gap[sl_a], np.where(a_fg, par, 0))
        gap[sl_b] = np.maximum(gap[sl_b], np.where(b_fg, par, 0))
    return Edges(cut_x, cut_y, fg, bg, gap)


def _one_sided(P: np.ndarray, cut: np.ndarray, axis: int):
    """Per-pixel derivative of P along `axis`, using only neighbours on the same surface.

    Returns (D, ok): D (H, W, 3) is the smaller-magnitude valid one-sided difference;
    ok is False where both neighbours are across a cut (or missing).
    """
    diff = np.diff(P, axis=axis)                  # P[k+1] - P[k]
    valid = ~cut & np.all(np.isfinite(diff), axis=-1)
    pad_shape = list(P.shape)
    pad_shape[axis] = 1
    nanpad = np.full(pad_shape, np.nan, P.dtype)
    fwd = np.concatenate([diff, nanpad], axis=axis)
    bwd = np.concatenate([nanpad, diff], axis=axis)
    vpad = np.zeros(pad_shape[:-1], bool)
    fwd_ok = np.concatenate([valid, vpad], axis=axis)
    bwd_ok = np.concatenate([vpad, valid], axis=axis)
    nf = np.where(fwd_ok, np.linalg.norm(np.nan_to_num(fwd), axis=-1), np.inf)
    nb = np.where(bwd_ok, np.linalg.norm(np.nan_to_num(bwd), axis=-1), np.inf)
    use_f = nf <= nb
    D = np.where(use_f[..., None], np.nan_to_num(fwd), np.nan_to_num(bwd))
    ok = np.isfinite(np.minimum(nf, nb))
    return D, ok


def matrix_to_quat_batch(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.empty((R.shape[0], 4), np.float64)
    c0 = t > 0
    c1 = ~c0 & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    c2 = ~c0 & ~c1 & (m[:, 1, 1] > m[:, 2, 2])
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(t[c0] + 1.0, 1e-12)) * 2
    q[c0] = np.stack([0.25 * s, (m[c0, 2, 1] - m[c0, 1, 2]) / s, (m[c0, 0, 2] - m[c0, 2, 0]) / s,
                      (m[c0, 1, 0] - m[c0, 0, 1]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m[c1, 0, 0] - m[c1, 1, 1] - m[c1, 2, 2], 1e-12)) * 2
    q[c1] = np.stack([(m[c1, 2, 1] - m[c1, 1, 2]) / s, 0.25 * s, (m[c1, 0, 1] + m[c1, 1, 0]) / s,
                      (m[c1, 0, 2] + m[c1, 2, 0]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m[c2, 1, 1] - m[c2, 0, 0] - m[c2, 2, 2], 1e-12)) * 2
    q[c2] = np.stack([(m[c2, 0, 2] - m[c2, 2, 0]) / s, (m[c2, 0, 1] + m[c2, 1, 0]) / s, 0.25 * s,
                      (m[c2, 1, 2] + m[c2, 2, 1]) / s], -1)
    s = np.sqrt(np.maximum(1.0 + m[c3, 2, 2] - m[c3, 0, 0] - m[c3, 1, 1], 1e-12)) * 2
    q[c3] = np.stack([(m[c3, 1, 0] - m[c3, 0, 1]) / s, (m[c3, 0, 2] + m[c3, 2, 0]) / s,
                      (m[c3, 1, 2] + m[c3, 2, 1]) / s, 0.25 * s], -1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def footprint_gaussians(points: np.ndarray, K: np.ndarray, cut_x: np.ndarray, cut_y: np.ndarray,
                        sigma_px: float, thickness: float, max_stretch: float = 30.0):
    """Gaussian covariance = the pixel's footprint on the reconstructed surface.

    With J = [dP/du, dP/dv] (3D step to the right / down neighbour on the same
    surface), Sigma = sigma^2 J J^T + (thin normal term). Rendered from the
    input camera every Gaussian then covers exactly its pixel; from a new
    viewpoint neighbouring Gaussians still overlap, so continuous surfaces stay
    closed (no cracks), while cut edges separate cleanly (no rubber sheets).

    Returns scales (N, 3) [sigma], quats (N, 4) [wxyz] for all H*W pixels.
    """
    H, W, _ = points.shape
    z = points[..., 2]
    Du, ok_u = _one_sided(points, cut_x, axis=1)
    Dv, ok_v = _one_sided(points, cut_y, axis=0)
    # Fronto-parallel fallback where a pixel has no same-surface neighbour (1-px structures).
    fu = np.zeros_like(points)
    fu[..., 0] = z / K[0, 0]
    fv = np.zeros_like(points)
    fv[..., 1] = z / K[1, 1]
    Du = np.where(ok_u[..., None], Du, fu)
    Dv = np.where(ok_v[..., None], Dv, fv)
    # Limit extreme stretching (grazing angles, depth noise).
    for D, ref in ((Du, z / K[0, 0]), (Dv, z / K[1, 1])):
        n = np.linalg.norm(D, axis=-1)
        lim = max_stretch * ref
        D *= np.minimum(1.0, lim / np.maximum(n, 1e-12))[..., None]
    Du = Du.reshape(-1, 3).astype(np.float64)
    Dv = Dv.reshape(-1, 3).astype(np.float64)
    n = np.cross(Du, Dv)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nn, 1e-20)
    in_plane = np.minimum(np.linalg.norm(Du, axis=1), np.linalg.norm(Dv, axis=1))
    t = thickness * sigma_px * in_plane
    cov = sigma_px ** 2 * (Du[:, :, None] * Du[:, None, :] + Dv[:, :, None] * Dv[:, None, :])
    cov += (t ** 2)[:, None, None] * n[:, :, None] * n[:, None, :]
    cov += (1e-6 * in_plane ** 2)[:, None, None] * np.eye(3)[None]
    evals, evecs = np.linalg.eigh(cov)
    evecs[np.linalg.det(evecs) < 0, :, 0] *= -1  # proper rotation
    scales = np.sqrt(np.maximum(evals, 1e-20))
    quats = matrix_to_quat_batch(evecs)
    return scales.astype(np.float32), quats.astype(np.float32)


def normals_from_points(points: np.ndarray, cut_x: np.ndarray, cut_y: np.ndarray) -> np.ndarray:
    """Camera-facing unit normals from same-surface finite differences."""
    Du, _ = _one_sided(points, cut_x, axis=1)
    Dv, _ = _one_sided(points, cut_y, axis=0)
    n = np.cross(Du, Dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-20)
    flip = np.sum(n * points, axis=-1) > 0
    n[flip] *= -1
    return n


def nearest_extrapolate(disp: np.ndarray, source: np.ndarray, targets: np.ndarray,
                        grad_x: Optional[np.ndarray] = None, grad_y: Optional[np.ndarray] = None):
    """Planar (first-order) extrapolation of disparity from the nearest `source` pixel.

    Disparity is an affine function of pixel position on any plane, so this
    continues walls, floors and ceilings exactly behind occluders and past the
    image border. Returns (values, source values) at `targets` (bool mask; NaN elsewhere).
    """
    if grad_x is None:
        grad_y, grad_x = np.gradient(disp)
    _, (iy, ix) = ndimage.distance_transform_edt(~source, return_indices=True)
    ty, tx = np.nonzero(targets)
    sy, sx = iy[ty, tx], ix[ty, tx]
    val = disp[sy, sx] + grad_x[sy, sx] * (tx - sx) + grad_y[sy, sx] * (ty - sy)
    out = np.full(disp.shape, np.nan, np.float64)
    src = np.full(disp.shape, np.nan, np.float64)
    out[ty, tx] = val
    src[ty, tx] = disp[sy, sx]
    return out, src
