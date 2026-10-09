"""Novel-view benchmark on the synthetic corridor (exact ground truth).

Compares, from ONE reference image:
  points        back-projected point cloud as isotropic splats (the naive "depth -> point cloud" baseline)
  surfels       per-pixel footprint Gaussians + depth-edge sharpening
  surfels+opt   ... + single-view 3DGS optimisation (the original design: "train 3DGS on the one image")
  surfels+hid   ... + hidden background layer and border extension (no optimisation)
  full          hidden layer + optimisation (this repository's default)

for two depth sources: ground-truth depth (isolates the 3DGS part) and Depth Anything V2
(relative depth; median-aligned to the ground truth because it has no metric scale).

Metrics on each novel view: PSNR / SSIM over the whole image, PSNR on pixels that ARE / are
NOT visible in the reference photo (seen / unseen = disoccluded + out of frame), and coverage
(alpha > 0.5). Each variant's Gaussians are saved (gs_*.npz) so `--eval-only` can re-score them.

    python -m bench.run_benchmark --out work/bench [--depth gt,dav2] [--steps 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bench.synthetic_corridor import default_camera, render, visible_from_reference
from ssplat.camera import look_at, translation_view
from ssplat.config import PipelineConfig
from ssplat.depth import _device
from ssplat.gaussians import GaussianSet
from ssplat.inpaint import make_inpainter
from ssplat.metrics import psnr, ssim
from ssplat.pipeline import naive_points, reconstruct
from ssplat.render import Renderer, to_uint8


def novel_views(median: float):
    tgt = np.array([0.0, 0.0, median])
    return {
        "x-0.1": translation_view([-0.1, 0, 0]), "x+0.1": translation_view([0.1, 0, 0]),
        "x-0.2": translation_view([-0.2, 0, 0]), "x+0.2": translation_view([0.2, 0, 0]),
        "x+0.3": translation_view([0.3, 0, 0]),
        "y-0.15": translation_view([0, -0.15, 0]), "y+0.15": translation_view([0, 0.15, 0]),
        "z+0.5": translation_view([0, 0, 0.5]), "z+1.0": translation_view([0, 0, 1.0]),
        "orbit-0.25": look_at(np.array([-0.25, 0, 0]), tgt),
    }


def ground_truth(out: Path, W: int, H: int):
    cache = out / "gt.npz"
    K = default_camera(W, H)
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return K, d["ref"], d["ref_depth"], d["views"].item()
    t = time.time()
    ref, ref_depth = render(np.eye(4), K, W, H, spp=3)
    med = float(np.median(ref_depth))
    views = {}
    for name, V in novel_views(med).items():
        img, _ = render(V, K, W, H, spp=3)
        vis = visible_from_reference(V, K, W, H, ref_depth)
        views[name] = {"V": V, "img": img, "vis": vis}
    np.savez_compressed(cache, ref=ref, ref_depth=ref_depth, views=np.array(views, dtype=object))
    print(f"[bench] ground truth rendered in {time.time() - t:.0f}s (median depth {med:.2f} m)", flush=True)
    return K, ref, ref_depth, views


def dav2_depth(img_u8: np.ndarray, gt_depth: np.ndarray, out: Path) -> np.ndarray:
    cache = out / "dav2_depth.npy"
    if cache.exists():
        return np.load(cache)
    from ssplat.config import DepthConfig
    from ssplat.depth import estimate_depth

    res = estimate_depth(img_u8, DepthConfig(model="dav2", fov_deg=70.0))
    d = res.depth * (np.median(gt_depth) / np.median(res.depth))  # oracle median scale
    np.save(cache, d)
    return d


def depth_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    r = np.maximum(pred / gt, gt / pred)
    return {"absrel": float(np.mean(np.abs(pred - gt) / gt)), "delta1": float(np.mean(r < 1.25))}


def evaluate(gs, K, W, H, views, dev) -> dict:
    r = Renderer(gs, dev)
    res = {}
    for name, v in views.items():
        o = r(v["V"], K, W, H)
        pred = torch.from_numpy(o["rgb"])
        gt = torch.from_numpy(v["img"])
        hidden = torch.from_numpy(~v["vis"])
        res[name] = {
            "psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
            "psnr_seen": psnr(pred, gt, ~hidden),
            "psnr_unseen": psnr(pred, gt, hidden) if hidden.any() else float("nan"),
            "unseen_frac": float(hidden.float().mean()),
            "coverage": float((o["alpha"] > 0.5).mean()),
        }
        res[name]["_img"] = o["rgb"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("work/bench"))
    ap.add_argument("--size", default="512x384")
    ap.add_argument("--depth", default="gt,dav2")
    ap.add_argument("--methods", default="points,surfels,surfels+opt,surfels+hid,full")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--inpainter", default="lama")
    ap.add_argument("--eval-only", action="store_true", help="Re-score saved gs_*.npz instead of reconstructing")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    W, H = (int(v) for v in args.size.split("x"))
    dev = _device("auto")
    K, ref, ref_depth, views = ground_truth(args.out, W, H)
    ref_u8 = to_uint8(ref)
    inpainter = None if args.eval_only else make_inpainter(args.inpainter)
    results = {"views": {k: {"unseen_frac": float((~v["vis"]).mean())} for k, v in views.items()}}
    strips = {}
    prev = {}
    if args.eval_only and (args.out / "results.json").exists():
        prev = json.loads((args.out / "results.json").read_text())
    for dsrc in args.depth.split(","):
        depth = ref_depth.astype(np.float32) if dsrc == "gt" else dav2_depth(ref_u8, ref_depth, args.out)
        results.setdefault("depth_error", {})[dsrc] = depth_errors(depth, ref_depth)
        for method in args.methods.split(","):
            t = time.time()
            cfg = PipelineConfig()
            cfg.optim.steps = args.steps if method in ("surfels+opt", "full") else 0
            cfg.occlusion.enabled = method in ("surfels+hid", "full")
            cfg.occlusion.inpainter = args.inpainter
            saved = args.out / f"gs_{dsrc}_{method.replace('+', '_')}.npz"
            if args.eval_only and saved.exists():
                gs = GaussianSet(**dict(np.load(saved)))
                info = {}
            elif method == "points":
                gs = naive_points(ref_u8.astype(np.float32) / 255.0, depth, K)
                info = {}
            else:
                rec = reconstruct(ref_u8, depth, K, cfg, inpainter=inpainter)
                gs, info = rec.gaussians, rec.info
            ev = evaluate(gs, K, W, H, views, dev)
            key = f"{dsrc}/{method}"
            imgs = {n: ev[n].pop("_img") for n in ev}
            strips[key] = {n: imgs[n] for n in ("x+0.3", "orbit-0.25", "z+1.0")}
            np.savez_compressed(args.out / f"gs_{dsrc}_{method.replace('+', '_')}.npz",
                                **{f: getattr(gs, f) for f in ("means", "quats", "scales", "opacities", "colors", "layer")})
            mean = {m: float(np.nanmean([ev[n][m] for n in ev]))
                    for m in ("psnr", "ssim", "psnr_seen", "psnr_unseen", "coverage")}
            if args.eval_only:
                info = {"optim": prev.get(key, {}).get("optim")}
            results[key] = {"mean": mean, "per_view": ev, "seconds": round(time.time() - t, 1),
                            "optim": info.get("optim"), "num_gaussians": len(gs)}
            print(f"[bench] {key:22s} PSNR {mean['psnr']:.2f}  SSIM {mean['ssim']:.4f}  seen {mean['psnr_seen']:.2f}  "
                  f"PSNR(unseen) {mean['psnr_unseen']:.2f}  coverage {mean['coverage']:.4f}  "
                  f"({time.time() - t:.0f}s)", flush=True)
            (args.out / "results.json").write_text(json.dumps(results, indent=2))
    # Visual comparison: rows = methods, columns = GT + views
    from PIL import Image

    for dsrc in args.depth.split(","):
        rows = []
        gt_row = np.concatenate([views[n]["img"] for n in ("x+0.3", "orbit-0.25", "z+1.0")], 1)
        rows.append(gt_row)
        for method in args.methods.split(","):
            rows.append(np.concatenate([strips[f"{dsrc}/{method}"][n] for n in ("x+0.3", "orbit-0.25", "z+1.0")], 1))
        Image.fromarray(to_uint8(np.concatenate(rows, 0))).save(args.out / f"compare_{dsrc}.jpg", quality=92)
    print(json.dumps({k: v["mean"] for k, v in results.items() if isinstance(v, dict) and "mean" in v}, indent=2))


if __name__ == "__main__":
    main()
