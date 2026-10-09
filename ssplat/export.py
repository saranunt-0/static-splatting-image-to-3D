"""Export: standard 3DGS .ply (+ .spz / .glb / .html via PlayCanvas splat-transform).

The world frame is the photo's camera (OpenCV axes), so the input viewpoint is
the origin looking down +z; `orientation` re-expresses it for other tools.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import ExportConfig
from .gaussians import GaussianSet
from .splats import Splats, write_ply

SPLAT_TRANSFORM_VERSION = "3.6.1"
NODE_DIR = Path.home() / ".cache" / "ssplat" / "node"

# Rows = new axes expressed in the OpenCV camera frame (x right, y down, z forward).
ORIENTATIONS = {
    "3dgs": np.eye(3),                                       # -Y up, +Z forward
    "y_up": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]),   # +Y up, -Z forward
    "z_up": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]),    # +Z up, +Y forward
}


def _log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def splat_transform_cmd() -> Optional[List[str]]:
    cli = NODE_DIR / "lib" / "node_modules" / "@playcanvas" / "splat-transform" / "bin" / "cli.mjs"
    if cli.exists():
        node = NODE_DIR / "bin" / "node"
        return [str(node) if node.exists() else "node", str(cli)]
    exe = shutil.which("splat-transform")
    if exe:
        return [exe]
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", f"@playcanvas/splat-transform@{SPLAT_TRANSFORM_VERSION}"]
    return None


def convert(src_ply: Path, dst: Path, extra: List[str] = ()) -> bool:
    cmd = splat_transform_cmd()
    if cmd is None:
        _log(f"skipping {dst.name}: splat-transform (Node.js >= 22) is not installed")
        return False
    full = cmd + ["-q", "-w", "-g", "cpu", str(src_ply), "-N", *extra, str(dst)]
    proc = subprocess.run(full, capture_output=True, text=True)
    if proc.returncode != 0:
        _log(f"failed to write {dst.name}:\n{proc.stderr.strip() or proc.stdout.strip()}")
        return False
    return True


def export_scene(gs: GaussianSet, out_dir: Path, cfg: ExportConfig, name: str = "scene",
                 meta: Optional[Dict] = None) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = gs.opacities >= cfg.min_opacity
    keep &= np.all(np.isfinite(gs.means), axis=1) & np.all(np.isfinite(gs.scales), axis=1)
    splats: Splats = gs.subset(keep).to_splats()
    if cfg.orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown orientation '{cfg.orientation}' ({sorted(ORIENTATIONS)})")
    canonical = splats
    user = splats if cfg.orientation == "3dgs" else splats.transformed(ORIENTATIONS[cfg.orientation])
    files: Dict[str, str] = {"ply": str(write_ply(user, out_dir / f"{name}.ply"))}
    conv_src = Path(files["ply"])
    tmp = None
    if user is not canonical:  # splat-transform and PlayCanvas expect the 3DGS convention
        tmp = write_ply(canonical, out_dir / f".{name}_3dgs_tmp.ply")
        conv_src = tmp
    for fmt in cfg.formats:
        if fmt == "ply":
            continue
        dst = {"spz": out_dir / f"{name}.spz", "glb": out_dir / f"{name}.glb",
               "html": out_dir / f"{name}_viewer.html"}.get(fmt)
        if dst is None:
            _log(f"unknown format '{fmt}', skipped")
            continue
        if convert(conv_src, dst, ["-H", "0"] if fmt == "html" else []):
            files[fmt] = str(dst)
    if tmp is not None:
        tmp.unlink()
    stats = {
        "num_gaussians": int(keep.sum()),
        "removed_transparent": int((~keep).sum()),
        "orientation": cfg.orientation,
        "camera": "input photo at the origin; OpenCV axes before re-orientation",
        "files": {k: {"path": v, "size_mb": round(Path(v).stat().st_size / 2**20, 1)} for k, v in files.items()},
        **(meta or {}),
    }
    (out_dir / "export_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    for k, v in stats["files"].items():
        _log(f"{k:>4}: {v['path']} ({v['size_mb']} MB)")
    return stats
