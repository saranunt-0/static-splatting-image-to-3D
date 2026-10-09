"""Lifting, occlusion edges, hidden layer and export on scenes with known geometry (CPU, no models)."""

import numpy as np
import torch

from ssplat.camera import intrinsics_from_fov, look_at, translation_view
from ssplat.config import ExportConfig, LiftConfig, OcclusionConfig, PipelineConfig
from ssplat.export import ORIENTATIONS, export_scene
from ssplat.gaussians import GaussianSet
from ssplat.geometry import occlusion_edges, sharpen_depth_edges
from ssplat.inpaint import TeleaInpainter
from ssplat.lift import lift_surface
from ssplat.occlusion import build_hidden_layer
from ssplat.render import Renderer
from ssplat.run import apply_overrides
from ssplat.splats import read_ply

W, H = 96, 72
K = intrinsics_from_fov(W, H, 60.0)


def _texture():
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([0.5 + 0.4 * np.sin(u / 5.0), 0.5 + 0.4 * np.cos(v / 7.0), 0.5 + 0.3 * np.sin((u + v) / 9.0)], -1)


def _step_depth(near=1.0, far=3.0):
    d = np.full((H, W), far, np.float32)
    d[24:48, 30:60] = near  # a box in front of a wall
    return d


def test_look_at_identity():
    assert np.allclose(look_at(np.zeros(3), np.array([0, 0, 5.0])), np.eye(4))


def test_edges_follow_parallax_physics():
    d = _step_depth()
    b = 0.1
    e = occlusion_edges(d, K[0, 0], b, parallax_px=1.0, min_ratio=0.03)
    assert e.fg[24:48, 30].all() and e.bg[24:48, 29].all()        # left side of the box
    assert not e.fg[:20].any() and not e.bg[:20].any()            # flat wall: no edges
    expected = K[0, 0] * b * (1 / 1.0 - 1 / 3.0)
    assert np.isclose(e.gap[30, 30], expected, rtol=1e-5)
    # The same jump is ignored when the expected camera motion cannot reveal a full pixel.
    e_small = occlusion_edges(d, K[0, 0], 1.0 / (K[0, 0] * 2), parallax_px=1.0, min_ratio=0.03)
    assert not e_small.fg.any()


def test_steep_smooth_floor_is_not_an_edge():
    # Floor plane 1.5 m below the camera: depth = f*h / (v - cy) below the horizon.
    v = np.arange(H)[:, None] + 0.5 - K[1, 2]
    d = np.where(v > 0.5, K[1, 1] * 1.5 / np.maximum(v, 0.5), 50.0).repeat(W, 1).astype(np.float32)
    d = np.minimum(d, 50.0)
    e = occlusion_edges(d, K[0, 0], 0.2, 1.0, 0.03)
    assert not e.fg[H // 2 + 3:].any()


def test_sharpen_removes_flying_pixels():
    d = _step_depth()
    d[24:48, 29] = 2.0  # blurred edge column halfway between box and wall
    s = sharpen_depth_edges(d, K[0, 0], 0.1, 1.0, 0.03)
    assert set(np.unique(np.round(s[30:40, 27:33], 3))) <= {1.0, 3.0}


def _render(gs, V=np.eye(4)):
    return Renderer(gs, torch.device("cpu"))(V, K, W, H)


def test_lift_reproduces_input_view_without_holes():
    rgb = _texture().astype(np.float32)
    d = _step_depth()
    e = occlusion_edges(d, K[0, 0], 0.1, 1.0, 0.03)
    gs = lift_surface(rgb, d, K, e, LiftConfig())
    out = _render(gs)
    assert out["alpha"].min() > 0.97
    err = np.abs(out["rgb"] - rgb).mean()
    assert err < 0.03, err
    assert np.allclose(out["depth"][30:40, 40:50], 1.0, rtol=0.02)


def test_tilted_plane_stays_closed_from_new_viewpoint():
    # Wall receding to the right (corridor-like): surfels must overlap, no cracks.
    u = np.arange(W)[None, :] + 0.5 - K[0, 2]
    x_over_z = u / K[0, 0]
    d = (1.0 / (0.6 - 0.5 * x_over_z)).repeat(H, 0).astype(np.float32)  # plane z = 1/(a - b x/z)
    e = occlusion_edges(d, K[0, 0], 0.1, 1.0, 0.03)
    gs = lift_surface(_texture().astype(np.float32), d, K, e, LiftConfig())
    out = _render(gs, translation_view([0.08, 0.0, 0.1]))
    inner = out["alpha"][8:-8, 8:-8]
    assert inner.min() > 0.9, inner.min()


def test_hidden_layer_continues_background_behind_box():
    rgb = (_texture() * 255).astype(np.uint8)
    d = np.full((H, W), 3.0, np.float32)
    d[10:62, 20:80] = 1.0  # large box: the strips of opposite edges do not meet
    cfg = OcclusionConfig(max_baseline=0.05)
    layer = build_hidden_layer(rgb, d, K, np.zeros((H, W), bool), cfg, TeleaInpainter(), median_depth=3.0)
    px, py = layer.pad
    band = layer.band[py:py + H, px:px + W]
    assert band.any() and not band[d > 2].any()            # strip lies on the box side only
    hd = layer.depth[py:py + H, px:px + W][band]
    assert np.allclose(hd, 3.0, rtol=0.05)                 # hidden surface = the wall behind
    gap = K[0, 0] * cfg.max_baseline * 3.0 * (1 - 1 / 3.0)
    width = band[36, 20:50].sum()   # strip behind the box's left edge
    assert gap <= width <= gap + cfg.band_margin_px + 2
    assert layer.border.sum() > 0 and px > 0 and py > 0


def test_export_roundtrip_and_orientations(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    q = rng.normal(size=(n, 4)).astype(np.float32)
    gs = GaussianSet(rng.normal(size=(n, 3)).astype(np.float32), q / np.linalg.norm(q, axis=1, keepdims=True),
                     rng.uniform(0.01, 0.1, (n, 3)).astype(np.float32), rng.uniform(0.1, 0.9, n).astype(np.float32),
                     rng.uniform(0, 1, (n, 3)).astype(np.float32))
    stats = export_scene(gs, tmp_path, ExportConfig(formats=["ply"], min_opacity=0.0))
    back = GaussianSet.from_splats(read_ply(stats["files"]["ply"]["path"]))
    assert np.allclose(back.means, gs.means, atol=1e-6)
    assert np.allclose(back.colors, gs.colors, atol=1e-5)
    assert np.allclose(back.opacities, gs.opacities, atol=1e-5)
    for R in ORIENTATIONS.values():
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    up = ORIENTATIONS["z_up"] @ np.array([0, -1.0, 0])     # camera "up" (-y in OpenCV)
    assert np.allclose(up, [0, 0, 1])
    assert np.allclose(ORIENTATIONS["y_up"] @ np.array([0, -1.0, 0]), [0, 1, 0])


def test_config_overrides():
    cfg = apply_overrides(PipelineConfig(), ["depth.model=dav2", "optim.steps=5", "method=sharp",
                                             "render.trajectories=swing,dolly", "occlusion.enabled=false"])
    assert cfg.depth.model == "dav2" and cfg.optim.steps == 5 and cfg.method == "sharp"
    assert cfg.render.trajectories == ["swing", "dolly"] and cfg.occlusion.enabled is False
