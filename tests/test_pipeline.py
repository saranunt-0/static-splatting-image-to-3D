"""End-to-end: CLI on a tiny synthetic photo with a given depth map (no network downloads)."""

import json

import numpy as np
from PIL import Image

from ssplat.config import PipelineConfig
from ssplat.pipeline import reconstruct
from ssplat.camera import intrinsics_from_fov
from ssplat.inpaint import TeleaInpainter
from ssplat.run import main


def _scene(W=64, H=48):
    v, u = np.mgrid[0:H, 0:W]
    img = np.stack([(u * 4) % 256, (v * 5) % 256, ((u + v) * 3) % 256], -1).astype(np.uint8)
    depth = np.full((H, W), 4.0, np.float32)
    depth[12:36, 20:40] = 1.5
    return img, depth


def test_reconstruct_layers_and_refinement():
    img, depth = _scene()
    K = intrinsics_from_fov(64, 48, 60.0)
    cfg = PipelineConfig()
    cfg.optim.steps = 40
    rec = reconstruct(img, depth, K, cfg, inpainter=TeleaInpainter())
    gs = rec.gaussians
    assert (gs.layer == 0).sum() == 64 * 48
    assert (gs.layer == 1).sum() > 0
    opt = rec.info["optim"]
    # The refinement minimises L1 + D-SSIM (PSNR can dip a little on this synthetic sawtooth texture).
    assert opt["l1_input_view_after"] < 0.8 * opt["l1_input_view_before"]
    assert opt["median_rel_depth_shift"] < 0.01  # geometry stays anchored
    hidden = gs.means[gs.layer == 1]
    assert np.isfinite(hidden).all() and (hidden[:, 2] > 0).all()


def test_cli_end_to_end(tmp_path):
    img, depth = _scene()
    Image.fromarray(img).save(tmp_path / "photo.png")
    np.save(tmp_path / "depth.npy", depth)
    work = tmp_path / "work"
    main(["--input", str(tmp_path / "photo.png"), "--work", str(work), "--preset", "fast",
          "--set", "depth.model=file", "--set", f"depth.depth_file={tmp_path / 'depth.npy'}",
          "--set", "depth.fov_deg=60", "--set", "occlusion.inpainter=telea", "--set", "optim.steps=3",
          "--set", "render.video=false", "--set", "export.formats=ply"])
    summary = json.loads((work / "summary.json").read_text())
    assert summary["depth"]["model"] == "file"
    assert summary["num_gaussians"]["visible"] == 64 * 48
    assert (work / "export" / "scene.ply").exists() and (work / "preview_grid.png").exists()
    assert (work / "debug" / "hidden_layer.png").exists()
