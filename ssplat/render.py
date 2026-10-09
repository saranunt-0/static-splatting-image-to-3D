"""Render novel views and preview videos of a GaussianSet."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .camera import trajectory
from .config import RenderConfig
from .gaussians import GaussianSet
from .rasterize import Camera, rasterize


class Renderer:
    def __init__(self, gs: GaussianSet, device: torch.device):
        t = gs.to_torch(device)
        self.t = t
        self.device = device

    @torch.no_grad()
    def __call__(self, viewmat: np.ndarray, K: np.ndarray, width: int, height: int,
                 background=(0.0, 0.0, 0.0)) -> Dict[str, np.ndarray]:
        cam = Camera(torch.tensor(K, dtype=torch.float32), torch.tensor(viewmat, dtype=torch.float32),
                     width, height).to(self.device)
        t = self.t
        out = rasterize(t["means"], t["quats"], t["scales"], t["opacities"], t["colors"].clamp_min(0), cam,
                        background=torch.tensor(background, device=self.device))
        a = out["alpha"]
        return {"rgb": out["rgb"].clamp(0, 1).cpu().numpy(), "alpha": a.cpu().numpy(),
                "depth": (out["depth"] / a.clamp_min(1e-6)).cpu().numpy()}


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def write_video(frames: List[np.ndarray], path: Path, fps: int) -> Path:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=2,
                            ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
        for f in frames:
            h, wd = f.shape[:2]
            w.append_data(f[: h - h % 2, : wd - wd % 2])
    return path


def render_videos(gs: GaussianSet, K: np.ndarray, width: int, height: int, median_depth: float,
                  cfg: RenderConfig, out_dir: Path, device: torch.device, name: str = "scene",
                  log=print) -> Dict[str, str]:
    r = Renderer(gs, device)
    Ks = K.copy()
    Ks[:2] *= cfg.scale
    w, h = int(round(width * cfg.scale)), int(round(height * cfg.scale))
    files = {}
    for kind in cfg.trajectories:
        views = trajectory(kind, cfg.frames, median_depth, cfg.motion, cfg.dolly)
        frames = [to_uint8(r(v, Ks, w, h)["rgb"]) for v in views]
        p = write_video(frames, out_dir / f"{name}_{kind}.mp4", cfg.fps)
        files[kind] = str(p)
        log(f"[render] {p}")
    return files


def preview_grid(gs: GaussianSet, K: np.ndarray, width: int, height: int, median_depth: float,
                 motion: float, device: torch.device, path: Path, extra_views: Optional[list] = None) -> Path:
    """3x3 grid of views on a plane around the input camera (centre = input view)."""
    from PIL import Image

    from .camera import look_at

    r = Renderer(gs, device)
    b = motion * median_depth
    target = np.array([0.0, 0.0, median_depth])
    tiles = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eye = np.array([dx * b, dy * b, 0.0])
            V = look_at(eye, target) if (dx or dy) else np.eye(4)
            tiles.append(to_uint8(r(V, K, width, height)["rgb"]))
    grid = np.concatenate([np.concatenate(tiles[i * 3:(i + 1) * 3], axis=1) for i in range(3)], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)
    return path
