"""Single image -> 3D Gaussian Splatting scene.

Examples::

    python -m ssplat.run --input corridor.jpg --work work/corridor
    python -m ssplat.run --input corridor.jpg --work work/corridor --preset fast \\
        --set depth.model=dav2 --set depth.fov_deg=65
    python -m ssplat.run --input corridor.jpg --work work/corridor_sharp --set method=sharp

Outputs (in --work): export/scene.ply (+ viewer .html), videos/*.mp4,
preview_grid.png (3x3 novel views), debug/ (depth, edges, hidden layer),
gaussians.npz (camera-frame scene for re-rendering) and summary.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import PRESETS, PipelineConfig, from_preset


def _coerce(current: Any, text: str) -> Any:
    if isinstance(current, bool):
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"Expected a boolean, got '{text}'")
    if isinstance(current, int):
        return int(text.replace("_", ""))
    if isinstance(current, float):
        return float(text)
    if isinstance(current, list):
        return [t.strip() for t in text.split(",") if t.strip()]
    return text


def apply_overrides(cfg: PipelineConfig, overrides: List[str]) -> PipelineConfig:
    for item in overrides:
        key, _, value = item.partition("=")
        section, _, fld = key.strip().partition(".")
        if not fld and hasattr(cfg, section) and not dataclasses.is_dataclass(getattr(cfg, section)):
            setattr(cfg, section, _coerce(getattr(cfg, section), value.strip()))
            continue
        target = getattr(cfg, section, None)
        if target is None or not dataclasses.is_dataclass(target) or not hasattr(target, fld):
            raise KeyError(f"Unknown option '{key}'. Format: section.field=value, e.g. optim.steps=300")
        setattr(target, fld, _coerce(getattr(target, fld), value.strip()))
    return cfg


def load_image(path: Path, max_long_edge: int) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        s = max_long_edge / max(im.size) if max_long_edge > 0 else 1.0
        if s < 1.0:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        return np.asarray(im).copy()


def _save_png(path: Path, img: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def _colorize_disp(depth: np.ndarray) -> np.ndarray:
    import matplotlib

    d = 1.0 / depth
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    x = np.clip((d - lo) / max(hi - lo, 1e-12), 0, 1)
    return (matplotlib.colormaps["turbo"](x)[..., :3] * 255).astype(np.uint8)


def run_layered(cfg: PipelineConfig, image_path: Path, work: Path) -> Dict:
    from .camera import fov_from_exif
    from .depth import DepthResult, _device, estimate_depth
    from .export import export_scene
    from .pipeline import reconstruct
    from .render import preview_grid, render_videos

    summary: Dict[str, Any] = {"input": str(image_path), "method": "layered"}
    t0 = time.time()
    img = load_image(image_path, cfg.lift.max_long_edge)
    H, W = img.shape[:2]
    summary["resolution"] = [W, H]

    # ---- depth (cached: re-running with other lift/optim settings skips the network)
    cache, key_file = work / "depth.npz", work / "depth_key.json"
    key = json.dumps({**dataclasses.asdict(cfg.depth), "size": [W, H]}, sort_keys=True)
    if cache.exists() and key_file.exists() and key_file.read_text() == key:
        res = DepthResult.load(cache)
        print(f"[depth] reusing {cache}", flush=True)
    else:
        res = estimate_depth(img, cfg.depth, fov_from_exif(image_path), cfg.optim.device)
        res.save(cache)
        key_file.write_text(key)
    summary["depth"] = {"model": res.model, "metric": res.metric,
                        "fov_x_deg": round(float(np.degrees(2 * np.arctan(0.5 * W / res.K[0, 0]))), 2)}
    summary["timing_s"] = {"depth": round(time.time() - t0, 1)}

    # ---- lift + hidden layer + single-view refinement
    t1 = time.time()
    rec = reconstruct(img, res.depth, res.K, cfg, sky=res.sky)
    summary.update(rec.info)
    summary["timing_s"]["reconstruct"] = round(time.time() - t1, 1)
    gs, K, median = rec.gaussians, rec.K, rec.median_depth
    np.savez_compressed(work / "gaussians.npz", K=K, width=W, height=H, median_depth=median,
                        **{f.name: getattr(gs, f.name) for f in dataclasses.fields(gs)})
    dbg = work / "debug"
    _save_png(dbg / "input.png", img)
    _save_png(dbg / "disparity.png", _colorize_disp(rec.depth))
    if rec.hidden is not None:
        hid = rec.hidden
        ov = img.copy()
        ov[hid.edges.fg] = [255, 0, 0]
        ov[hid.edges.bg] = [0, 255, 255]
        _save_png(dbg / "occlusion_edges.png", ov)
        vis = hid.rgb.copy()
        vis[~hid.mask] = (vis[~hid.mask] * 0.35).astype(np.uint8)
        _save_png(dbg / "hidden_layer.png", vis)
        _save_png(dbg / "inpaint_mask.png", (hid.inpaint_mask * 255).astype(np.uint8))

    # ---- export + previews
    t3 = time.time()
    summary["export"] = export_scene(gs, work / "export", cfg.export,
                                     meta={"median_depth": median, "metric": res.metric,
                                           "K": K.tolist(), "image_size": [W, H]})
    dev = _device(cfg.optim.device)
    preview_grid(gs, K, W, H, median, cfg.render.motion, dev, work / "preview_grid.png")
    if cfg.render.video:
        summary["videos"] = render_videos(gs, K, W, H, median, cfg.render, work / "videos", dev)
    summary["timing_s"]["export_render"] = round(time.time() - t3, 1)
    summary["timing_s"]["total"] = round(time.time() - t0, 1)
    return summary


def run_sharp(cfg: PipelineConfig, image_path: Path, work: Path) -> Dict:
    """Apple SHARP (feed-forward, research-only licence) + this repo's export and previews."""
    from plyfile import PlyData

    from .depth import _device
    from .export import export_scene
    from .gaussians import GaussianSet
    from .render import preview_grid, render_videos
    from .splats import read_ply

    exe = shutil.which("sharp")
    if exe is None:
        raise RuntimeError("SHARP is not installed: python -m ssplat.setup_env --sharp "
                           "(research-only licence, see docs/RESEARCH.md)")
    out = work / "sharp"
    t0 = time.time()
    dev = _device(cfg.optim.device)
    cmd = [exe, "predict", "-i", str(image_path), "-o", str(out), "--device", dev.type]
    if cfg.sharp_checkpoint:
        cmd += ["-c", cfg.sharp_checkpoint]
    subprocess.run(cmd, check=True)
    ply = out / f"{image_path.stem}.ply"
    gs = GaussianSet.from_splats(read_ply(ply))
    data = PlyData.read(str(ply))
    K = np.asarray(data["intrinsic"].data["intrinsic"], dtype=np.float64).reshape(3, 3)
    W, H = (int(v) for v in data["image_size"].data["image_size"])
    median = float(np.median(gs.means[:, 2]))
    summary = {"input": str(image_path), "method": "sharp", "num_gaussians": len(gs),
               "median_depth": median, "resolution": [W, H], "timing_s": {"predict": round(time.time() - t0, 1)}}
    summary["export"] = export_scene(gs, work / "export", cfg.export,
                                     meta={"median_depth": median, "metric": True, "K": K.tolist(),
                                           "image_size": [W, H], "licence": "Apple ML Research Model (research only)"})
    s = min(1.0, cfg.lift.max_long_edge / max(W, H))
    Ks = K.copy()
    Ks[:2] *= s
    w, h = round(W * s), round(H * s)
    preview_grid(gs, Ks, w, h, median, cfg.render.motion, dev, work / "preview_grid.png")
    if cfg.render.video:
        summary["videos"] = render_videos(gs, Ks, w, h, median, cfg.render, work / "videos", dev)
    return summary


def run(cfg: PipelineConfig, image_path: Path, work: Path) -> Dict:
    work.mkdir(parents=True, exist_ok=True)
    cfg.save(work / "config.json")
    if cfg.method == "layered":
        summary = run_layered(cfg, image_path, work)
    elif cfg.method == "sharp":
        summary = run_sharp(cfg, image_path, work)
    else:
        raise ValueError(f"Unknown method '{cfg.method}' (layered, sharp)")
    (work / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("export",)}, indent=2, default=str))
    return summary


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Input photo (jpg/png/...)")
    ap.add_argument("--work", type=Path, required=True, help="Output directory")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="quality")
    ap.add_argument("--config", type=Path, help="JSON config (overrides the preset)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                    help="Override one option, e.g. --set depth.model=dav2 (repeatable)")
    args = ap.parse_args(argv)
    cfg = PipelineConfig.load(args.config) if args.config else from_preset(args.preset)
    cfg = apply_overrides(cfg, args.overrides)
    run(cfg, args.input, args.work)


if __name__ == "__main__":
    main()
