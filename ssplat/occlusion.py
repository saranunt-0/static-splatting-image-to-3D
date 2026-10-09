"""Hidden background layer: what is behind foreground edges and beyond the photo frame.

A single photo only shows the front-most surface. As soon as the camera moves,
two kinds of holes open up:

1. behind every occluding edge (a door frame, a pillar, a person), the strip of
   background it was hiding, and
2. beyond the image border.

This module builds a second layer of Gaussians for exactly those regions, in the
spirit of layered-depth "3D photo" methods (Shih et al. CVPR 2020, SLIDE ICCV
2021) and of the two-layer output of Apple SHARP (2025):

* Occlusion edges and their width come from physics, not a fixed threshold: an
  edge between disparities d1 > d2 opens f * b * (d1 - d2) pixels when the
  camera moves by b (`occlusion.max_baseline`). Only that strip is synthesised.
* Hidden depth = planar extrapolation of the background's disparity from the
  nearest trusted background pixel. Disparity is affine in pixel coordinates on
  any plane, so walls, floors and ceilings (most of a corridor) continue exactly.
* Hidden colour = LaMa inpainting, with the foreground near the edge masked out
  as well so the inpainter only copies background texture into the strip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import OcclusionConfig
from .geometry import Edges, nearest_extrapolate, occlusion_edges


def _log(msg: str) -> None:
    print(f"[occlusion] {msg}", flush=True)


@dataclass
class HiddenLayer:
    rgb: np.ndarray           # (Hc, Wc, 3) uint8 canvas with inpainted content
    depth: np.ndarray         # (Hc, Wc) float, NaN outside `mask`
    mask: np.ndarray          # (Hc, Wc) bool: pixels that become hidden-layer Gaussians
    band: np.ndarray          # (Hc, Wc) bool: part of mask behind foreground edges
    border: np.ndarray        # (Hc, Wc) bool: part of mask outside the original frame
    inpaint_mask: np.ndarray  # (Hc, Wc) bool: what the inpainter was asked to fill
    K: np.ndarray             # canvas intrinsics
    pad: tuple                # (pad_x, pad_y) in pixels
    edges: Edges              # occlusion edges of the input depth map

    def stats(self) -> dict:
        H, W = self.depth.shape
        h, w = H - 2 * self.pad[1], W - 2 * self.pad[0]
        return {"hidden_band_px": int(self.band.sum()), "border_px": int(self.border.sum()),
                "band_fraction_of_image": round(float(self.band.sum()) / (h * w), 4),
                "pad_xy": list(self.pad), "edge_px": int(self.edges.fg.sum())}  # before fragment filter


def _masked_smooth(values: np.ndarray, mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Box filter restricted to `mask` (normalised convolution)."""
    v = np.where(mask, values, 0.0)
    num = ndimage.uniform_filter(v, size=size, mode="nearest")
    den = ndimage.uniform_filter(mask.astype(np.float64), size=size, mode="nearest")
    out = values.copy()
    out[mask] = num[mask] / np.maximum(den[mask], 1e-12)
    return out


def build_hidden_layer(image: np.ndarray, depth: np.ndarray, K: np.ndarray, sky: np.ndarray,
                       cfg: OcclusionConfig, inpainter, median_depth: float) -> HiddenLayer:
    """image (H, W, 3) uint8; depth (H, W); returns the hidden layer on a padded canvas."""
    H, W = depth.shape
    f = float(K[0, 0])
    b = cfg.max_baseline * median_depth
    disp = 1.0 / depth.astype(np.float64)
    E = occlusion_edges(depth, f, b, cfg.edge_parallax_px, cfg.edge_max_angle_deg)
    max_extent = max(4, int(cfg.max_extent_frac * W))

    # ---- 1. strip behind foreground edges ---------------------------------------------
    fg_edges = E.fg
    if cfg.min_edge_px > 1 and fg_edges.any():
        lab, n = ndimage.label(fg_edges, structure=np.ones((3, 3)))
        sizes = ndimage.sum(fg_edges, lab, index=np.arange(1, n + 1))
        fg_edges = np.isin(lab, 1 + np.nonzero(sizes >= cfg.min_edge_px)[0])
    radius = np.where(fg_edges, np.minimum(np.ceil(E.gap) + cfg.band_margin_px, max_extent), 0.0)
    if fg_edges.any():
        dist_fg, (iy, ix) = ndimage.distance_transform_edt(~fg_edges, return_indices=True)
        r_near = radius[iy, ix]
        _, (by, bx) = ndimage.distance_transform_edt(~E.bg, return_indices=True)
        d_bg = disp[by, bx]
        front = disp > d_bg * (1 + cfg.edge_min_ratio)      # in front of the nearby background
        # Foreground around edges. It is also masked for the inpainter so that it copies
        # background (not foreground) texture into the strip ("ghosting" otherwise); not
        # wider, because large holes make inpainting blurrier.
        fg_near = front & (dist_fg <= 2 * r_near + 2) & ~sky
        band = fg_near & (dist_fg <= r_near)
    else:
        front = np.zeros((H, W), bool)
        fg_near = np.zeros((H, W), bool)
        band = np.zeros((H, W), bool)

    # ---- 2. canvas with border extension ------------------------------------------------
    if cfg.extend_border:
        # Moving sideways by b reveals f*b*disp pixels beyond the frame; looking back at the
        # scene centre (swing/circle trajectories) rotates the view by about b/median_depth.
        rot = f * b / median_depth
        pad_x = int(math.ceil(f * b * float(np.max(disp[:, [0, -1]])) + rot)) + cfg.band_margin_px
        pad_y = int(math.ceil(f * b * float(np.max(disp[[0, -1], :])) + rot)) + cfg.band_margin_px
        pad_x, pad_y = min(pad_x, max_extent), min(pad_y, max_extent)
    else:
        pad_x = pad_y = 0
    Hc, Wc = H + 2 * pad_y, W + 2 * pad_x
    inner = (slice(pad_y, pad_y + H), slice(pad_x, pad_x + W))

    def to_canvas(a, fill):
        out = np.full((Hc, Wc) + a.shape[2:], fill, dtype=a.dtype)
        out[inner] = a
        return out

    border = np.ones((Hc, Wc), bool)
    border[inner] = False
    band_c = to_canvas(band, False)
    targets = band_c | border

    # ---- 3. hidden depth: planar continuation of the background -------------------------
    # Background-only smoothing (normalised convolution): the slope used for extrapolation
    # must not see the foreground, or the hidden surface bends towards it.
    bgmask = ~ndimage.binary_dilation(E.fg | E.bg, iterations=2) & ~fg_near & ~sky
    wsum = ndimage.gaussian_filter(bgmask.astype(np.float64), 1.5, mode="nearest")
    smooth = ndimage.gaussian_filter(np.where(bgmask, disp, 0.0), 1.5, mode="nearest") / np.maximum(wsum, 1e-12)
    smooth = np.where(bgmask, smooth, disp)
    gy, gx = np.gradient(smooth)
    trusted = ndimage.binary_erosion(bgmask, iterations=2)  # gradient stencil inside the background
    if not trusted.any():
        trusted = bgmask if bgmask.any() else np.ones_like(sky)
    disp_c = np.nan_to_num(to_canvas(disp, np.nan), nan=1.0)
    gx_c, gy_c = to_canvas(gx, 0.0), to_canvas(gy, 0.0)
    trusted_c = to_canvas(trusted, False)
    # The strip behind an edge continues the surface on the far side of that edge, so its
    # sources exclude everything that is in front of nearby background (incl. the occluder).
    # Far / sky pixels are valid background too (constant depth: zero slope).
    sky_c = to_canvas(sky, False)
    gx_c[sky_c] = 0.0
    gy_c[sky_c] = 0.0
    sources = trusted_c | sky_c
    band_src = sources & ~to_canvas(front, False)
    if not band_src.any():
        band_src = sources
    ext_b, src_b = nearest_extrapolate(disp_c, band_src, band_c, gx_c, gy_c)
    # Beyond the frame, the nearest pixels at the image border are the right sources.
    ext_o, src_o = nearest_extrapolate(disp_c, sources, border, gx_c, gy_c)
    ext = np.where(band_c, ext_b, ext_o)
    src = np.where(band_c, src_b, src_o)
    ext = np.clip(ext, 0.5 * src, 2.0 * src)
    # Hidden surfaces must stay behind the foreground they are hidden by.
    behind = np.where(border, np.inf, disp_c) / (1 + cfg.edge_min_ratio)
    ext = np.where(band_c, np.minimum(ext, behind), ext)
    ext = _masked_smooth(np.nan_to_num(ext, nan=0.0), targets, size=5)
    ext = np.where(band_c, np.minimum(ext, behind), ext)
    ext = np.maximum(ext, 1.0 / (np.max(depth) * 1.5))
    hidden_depth = np.where(targets, 1.0 / ext, np.nan)

    # ---- 4. hidden colour: inpaint with the foreground near edges masked as well ---------
    inpaint_mask = targets | to_canvas(fg_near, False)
    canvas = to_canvas(image, 0)
    rgb = inpainter(canvas, inpaint_mask) if inpaint_mask.any() else canvas

    Kc = K.copy()
    Kc[0, 2] += pad_x
    Kc[1, 2] += pad_y
    layer = HiddenLayer(rgb, hidden_depth.astype(np.float32), targets, band_c, border, inpaint_mask,
                        Kc, (pad_x, pad_y), E)
    _log(f"{int(fg_edges.sum())} edge px, hidden strip {int(band.sum())} px "
         f"({100 * band.mean():.1f}% of image), border pad {pad_x}x{pad_y} px, inpainter {inpainter.name}")
    return layer
