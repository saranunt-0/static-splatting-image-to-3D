"""Procedural corridor with exact ground truth for single-image novel-view benchmarks.

A ray caster over axis-aligned boxes with world-space procedural textures and
view-independent (Lambertian) shading, so every viewpoint has an exact image and
depth map. The scene has what makes corridors hard: long perspective, grazing
floor/walls/ceiling, recessed doors (self-occlusion at the frames), a pillar, a
free-standing cabinet and a hanging sign (thin occluder with background around it).

World frame: OpenCV (x right, y down, z forward); the reference camera is at the
origin, 1.6 m above the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

FLOOR_Y, CEIL_Y = 1.6, -1.2        # camera height 1.6 m, ceiling height 2.8 m
WALL_L, WALL_R = -1.25, 1.25       # corridor width 2.5 m
Z_NEAR, Z_END = -3.0, 16.0
DOOR_TOP = -0.5                    # doors are 2.1 m high
RECESS = 0.14


@dataclass
class Box:
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    material: str


def _scene() -> List[Box]:
    T = 0.3  # wall / slab thickness
    boxes = [
        Box((WALL_L - 1, FLOOR_Y, Z_NEAR), (WALL_R + 1, FLOOR_Y + T, Z_END + T), "floor"),
        Box((WALL_L - 1, CEIL_Y - T, Z_NEAR), (WALL_R + 1, CEIL_Y, Z_END + T), "ceiling"),
        Box((WALL_L - 1, CEIL_Y - T, Z_END), (WALL_R + 1, FLOOR_Y + T, Z_END + T), "endwall"),
    ]
    # Walls with door openings: (side, z0, z1)
    doors = [("L", 2.6, 3.6), ("L", 8.0, 9.0), ("R", 5.4, 6.4), ("R", 11.0, 12.0)]
    for side in ("L", "R"):
        x0, x1 = (WALL_L - T, WALL_L) if side == "L" else (WALL_R, WALL_R + T)
        cuts = sorted((z0, z1) for s, z0, z1 in doors if s == side)
        z = Z_NEAR
        for z0, z1 in cuts:
            boxes.append(Box((x0, CEIL_Y, z), (x1, FLOOR_Y, z0), "wall"))
            boxes.append(Box((x0, CEIL_Y, z0), (x1, DOOR_TOP, z1), "wall"))  # lintel
            z = z1
        boxes.append(Box((x0, CEIL_Y, z), (x1, FLOOR_Y, Z_END), "wall"))
        for z0, z1 in cuts:  # door panel recessed behind the wall surface
            if side == "L":
                boxes.append(Box((WALL_L - RECESS - 0.05, DOOR_TOP, z0), (WALL_L - RECESS, FLOOR_Y, z1), "door"))
            else:
                boxes.append(Box((WALL_R + RECESS, DOOR_TOP, z0), (WALL_R + RECESS + 0.05, FLOOR_Y, z1), "door"))
    boxes += [
        Box((WALL_R - 0.32, CEIL_Y, 4.0), (WALL_R, FLOOR_Y, 4.45), "pillar"),
        Box((WALL_L, 0.85, 5.6), (WALL_L + 0.45, FLOOR_Y, 6.1), "cabinet"),
        Box((-0.45, -0.95, 7.0), (0.45, -0.62, 7.04), "sign"),
        Box((-0.02, CEIL_Y, 7.01), (0.02, -0.95, 7.03), "wire"),
    ]
    return boxes


BOXES = _scene()


def _hash(*a: np.ndarray) -> np.ndarray:
    h = np.zeros_like(a[0], dtype=np.float64)
    for i, v in enumerate(a):
        h = h + np.floor(v) * (127.1 + 311.7 * i)
    return np.modf(np.abs(np.sin(h) * 43758.5453))[0]


def _vnoise(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = _hash(iu, iv), _hash(iu + 1, iv)
    c, d = _hash(iu, iv + 1), _hash(iu + 1, iv + 1)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def _fbm(u, v, octaves=4):
    s, amp, f = 0.0, 0.5, 1.0
    for _ in range(octaves):
        s = s + amp * _vnoise(u * f, v * f)
        amp *= 0.5
        f *= 2.0
    return s


def _texture(mat: np.ndarray, p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Albedo for hit points p with face normals n and material ids."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    col = np.zeros_like(p)

    def put(m, rgb):
        col[m] = rgb[m] if rgb.ndim == 2 else rgb

    # floor: 0.6 m tiles, per-tile tint, grout, speckle
    m = mat == 0
    tu, tv = x / 0.6, z / 0.6
    grout = (np.minimum(np.modf(np.abs(tu))[0], 1 - np.modf(np.abs(tu))[0]) < 0.025) | \
            (np.minimum(np.modf(np.abs(tv))[0], 1 - np.modf(np.abs(tv))[0]) < 0.025)
    tint = 0.75 + 0.35 * _hash(tu, tv)
    speck = 0.85 + 0.3 * _fbm(x * 9, z * 9)
    base = np.stack([0.62 * tint * speck, 0.56 * tint * speck, 0.48 * tint * speck], -1)
    base[grout] = [0.25, 0.24, 0.23]
    put(m, base)
    # ceiling: panels + lights
    m = mat == 1
    pu, pv = x / 0.6, z / 0.6
    line = (np.minimum(np.modf(np.abs(pu))[0], 1 - np.modf(np.abs(pu))[0]) < 0.02) | \
           (np.minimum(np.modf(np.abs(pv))[0], 1 - np.modf(np.abs(pv))[0]) < 0.02)
    c = np.stack([0.86 + 0 * x, 0.86 + 0 * x, 0.84 + 0 * x], -1) * (0.95 + 0.1 * _fbm(x * 4, z * 4))[:, None]
    c[line] = [0.55, 0.55, 0.55]
    light = (np.abs(x) < 0.3) & (np.abs(np.mod(z - 1.0, 2.5) - 1.25) < 0.35)
    c[light] = [1.0, 0.98, 0.9]
    put(m, c)
    # walls: painted, wainscot, chair rail, baseboard, posters
    m = mat == 2
    wu = z
    paint = (0.9 + 0.12 * _fbm(wu * 3, y * 3))[:, None] * np.array([0.72, 0.78, 0.86])
    lower = y > 0.65
    paint[lower] = ((0.85 + 0.15 * _fbm(wu[lower] * 6, y[lower] * 6))[:, None] * np.array([0.42, 0.52, 0.47]))
    paint[(y > 0.6) & (y < 0.68)] = [0.55, 0.36, 0.22]
    paint[y > 1.48] = [0.2, 0.17, 0.15]
    side = np.sign(x)
    for (sd, z0, z1, hue) in ((-1, 0.6, 1.8, 0), (1, 1.2, 2.6, 1), (1, 7.0, 8.2, 2), (-1, 10.0, 11.0, 3), (-1, 5.0, 5.4, 4)):
        poster = (side == sd) & (z > z0) & (z < z1) & (y > -0.45) & (y < 0.35)
        if poster.any():
            k = _fbm(z[poster] * 5 + hue * 3, y[poster] * 5)
            stripes = (np.sin((z[poster] + y[poster]) * 18 + hue) > 0).astype(float)
            pal = np.array([[0.9, 0.3, 0.2], [0.2, 0.5, 0.9], [0.95, 0.8, 0.2], [0.3, 0.75, 0.35], [0.8, 0.3, 0.7]])
            a, b = pal[hue], pal[(hue + 2) % 5]
            paint[poster] = (stripes[:, None] * a + (1 - stripes[:, None]) * b) * (0.7 + 0.5 * k[:, None])
    put(m, paint)
    # doors: wood grain + handle
    m = mat == 3
    grain = 0.75 + 0.25 * np.sin(y * 40 + 6 * _fbm(y * 3, z * 20)) * 0.5 + 0.2 * _fbm(z * 30, y * 2)
    wood = grain[:, None] * np.array([0.55, 0.33, 0.17])
    zc = np.mod(z, 1000.0)
    handle = (np.abs(y - 0.55) < 0.04) & (np.abs(np.modf(zc)[0] - 0.85) < 0.06)
    wood[handle] = [0.85, 0.82, 0.7]
    put(m, wood)
    # pillar, cabinet, sign, wire, end wall
    put(mat == 4, np.array([0.8, 0.78, 0.74]) * (0.85 + 0.2 * _fbm(z * 8 + x * 8, y * 8))[:, None])
    m = mat == 5
    cab = np.array([0.25, 0.35, 0.6]) * (0.85 + 0.25 * _fbm(z * 10, y * 10))[:, None]
    cab[(np.abs(y - 1.1) < 0.02)] = [0.1, 0.1, 0.12]
    put(m, cab)
    m = mat == 6
    sg = np.tile(np.array([0.1, 0.55, 0.25]), (len(x), 1))
    letters = (np.abs(y + 0.785) < 0.07) & (np.sin(x * 60) > 0.2) & (np.abs(x) < 0.38)
    sg[letters] = [0.97, 0.97, 0.95]
    put(m, sg)
    put(mat == 7, np.array([0.15, 0.15, 0.15]))
    m = mat == 8
    ew = (0.9 + 0.1 * _fbm(x * 3, y * 3))[:, None] * np.array([0.8, 0.72, 0.6])
    win = (np.abs(x) < 0.7) & (y > -0.6) & (y < 0.6)
    ew[win] = np.array([0.55, 0.75, 0.95]) * (0.9 + 0.2 * _fbm(x[win] * 2, y[win] * 2))[:, None]
    frame = win & ((np.abs(x) > 0.66) | (np.abs(y) > 0.56) | (np.abs(x) < 0.02) | (np.abs(y) < 0.02))
    ew[frame] = [0.95, 0.95, 0.95]
    put(m, ew)
    return col


_MATS = {"floor": 0, "ceiling": 1, "wall": 2, "door": 3, "pillar": 4, "cabinet": 5, "sign": 6, "wire": 7,
         "endwall": 8}
_EMISSIVE = None


def cast(origins: np.ndarray, dirs: np.ndarray):
    """Nearest box hit for each ray: (t, point, normal, material id)."""
    n = dirs.shape[0]
    best_t = np.full(n, np.inf)
    best_n = np.zeros((n, 3))
    best_m = np.full(n, -1)
    inv = 1.0 / np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    for box in BOXES:
        lo, hi = np.array(box.lo), np.array(box.hi)
        t1 = (lo - origins) * inv
        t2 = (hi - origins) * inv
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        t_enter = tmin.max(1)
        t_exit = tmax.min(1)
        hit = (t_enter <= t_exit) & (t_enter > 1e-6) & (t_enter < best_t)
        if not hit.any():
            continue
        axis = tmin[hit].argmax(1)
        nrm = np.zeros((int(hit.sum()), 3))
        nrm[np.arange(len(axis)), axis] = -np.sign(dirs[hit][np.arange(len(axis)), axis])
        best_t[hit] = t_enter[hit]
        best_n[hit] = nrm
        best_m[hit] = _MATS[box.material]
    p = origins + dirs * best_t[:, None]
    return best_t, p, best_n, best_m


def shade(p, n, m) -> np.ndarray:
    alb = _texture(m, p, n)
    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    lam = 0.55 + 0.45 * np.clip(-(n @ light), 0, 1) + 0.12 * np.clip(n @ np.array([0.0, 0.0, -1.0]), 0, 1)
    # gentle distance falloff along the corridor (view independent: depends on z only)
    fall = 1.0 / (1.0 + 0.012 * np.maximum(p[:, 2], 0))
    out = alb * (lam * fall)[:, None]
    emissive = (m == 1) & (np.abs(p[:, 0]) < 0.3) & (np.abs(np.mod(p[:, 2] - 1.0, 2.5) - 1.25) < 0.35)
    out[emissive] = alb[emissive]
    win = (m == 8) & (np.abs(p[:, 0]) < 0.7) & (p[:, 1] > -0.6) & (p[:, 1] < 0.6)
    out[win] = alb[win]
    return np.clip(out, 0, 1)


def render(viewmat: np.ndarray, K: np.ndarray, width: int, height: int, spp: int = 3):
    """Ground-truth image (supersampled spp x spp) and z-depth (pixel centres)."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    c2w_R = R.T
    origin = -R.T @ t
    img = np.zeros((height * width, 3))
    offs = [(i + 0.5) / spp for i in range(spp)]
    jj, ii = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    for oy in offs:
        for ox in offs:
            u = ii.reshape(-1) + ox
            v = jj.reshape(-1) + oy
            d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
            d = d_cam @ c2w_R.T
            _, p, n, m = cast(np.broadcast_to(origin, d.shape), d)
            img += shade(p, n, m)
    img /= spp * spp
    u = ii.reshape(-1) + 0.5
    v = jj.reshape(-1) + 0.5
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    tt, p, _, _ = cast(np.broadcast_to(origin, d_cam.shape), d_cam @ c2w_R.T)
    depth = tt  # d_cam has unit z, so ray parameter t equals z-depth in the camera frame
    return img.reshape(height, width, 3).astype(np.float32), depth.reshape(height, width).astype(np.float32)


def default_camera(width: int = 512, height: int = 384, fov_x: float = 70.0) -> np.ndarray:
    fx = 0.5 * width / math.tan(math.radians(fov_x) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1.0]])


def visible_from_reference(viewmat: np.ndarray, K: np.ndarray, width: int, height: int,
                           ref_depth: np.ndarray, tol: float = 0.02) -> np.ndarray:
    """Mask of novel-view pixels whose surface point is visible in the reference view.

    The complement is the disoccluded / out-of-frame region that a single photo cannot show.
    """
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    jj, ii = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    u, v = ii.reshape(-1) + 0.5, jj.reshape(-1) + 0.5
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    _, p, _, _ = cast(np.broadcast_to(-R.T @ t, d_cam.shape), d_cam @ R)
    # project into the reference camera (identity pose)
    z = p[:, 2]
    pu = K[0, 0] * p[:, 0] / z + K[0, 2]
    pv = K[1, 1] * p[:, 1] / z + K[1, 2]
    inside = (z > 0) & (pu >= 0) & (pu < width) & (pv >= 0) & (pv < height)
    vis = np.zeros(len(z), bool)
    iu = np.clip(pu.astype(int), 0, width - 1)
    iv = np.clip(pv.astype(int), 0, height - 1)
    vis[inside] = np.abs(ref_depth[iv[inside], iu[inside]] - z[inside]) <= tol * z[inside]
    return vis.reshape(height, width)
