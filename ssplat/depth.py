"""Monocular geometry backends: image -> depth map + intrinsics (+ normals).

All backends return a `DepthResult` at the input image's resolution, with
depth as z-distance along the optical axis (OpenCV camera frame).
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .camera import intrinsics_from_fov
from .config import DepthConfig

CACHE_DIR = Path.home() / ".cache" / "ssplat"
DAV2_URL = ("https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/"
            "depth_anything_v2_vitl_dynamic.onnx")


def _log(msg: str) -> None:
    print(f"[depth] {msg}", flush=True)


@dataclass
class DepthResult:
    depth: np.ndarray            # (H, W) float32, z-depth; finite and > 0 everywhere
    K: np.ndarray                # (3, 3) intrinsics in pixels
    metric: bool                 # True if depth is in metres
    model: str
    normal: Optional[np.ndarray] = None  # (H, W, 3) camera-frame normals, if the model predicts them
    sky: Optional[np.ndarray] = None     # (H, W) bool: pixels at "infinity" (clamped far)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, depth=self.depth, K=self.K, metric=self.metric, model=self.model,
                            normal=self.normal if self.normal is not None else np.zeros(0),
                            sky=self.sky if self.sky is not None else np.zeros(0))

    @classmethod
    def load(cls, path: Path) -> "DepthResult":
        d = np.load(path, allow_pickle=False)
        normal = d["normal"] if d["normal"].size else None
        sky = d["sky"] if d["sky"].size else None
        return cls(d["depth"], d["K"], bool(d["metric"]), str(d["model"]), normal, sky)


def download(url: str, name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / "weights" / name
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    _log(f"downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def _device(name: str = "auto"):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _finalize(depth: np.ndarray, valid: np.ndarray, cfg: DepthConfig) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid pixels with the far clamp; returns (depth, sky_mask)."""
    good = valid & np.isfinite(depth) & (depth > 0)
    med = float(np.median(depth[good])) if good.any() else 1.0
    far = cfg.max_depth_ratio * med
    sky = ~good | (depth > far)
    depth = np.where(sky, far, depth).astype(np.float32)
    return depth, sky


# ----------------------------------------------------------------------------- MoGe

def _moge(image: np.ndarray, cfg: DepthConfig, version: str, device: str) -> DepthResult:
    import torch

    try:
        if version == "v2":
            from moge.model.v2 import MoGeModel
        else:
            from moge.model.v3 import MoGeModel
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "MoGe is not installed. Run `python -m ssplat.setup_env --moge` (or "
            "`pip install git+https://github.com/microsoft/MoGe.git`), or use --set depth.model=dav2"
        ) from e
    dev = _device(device)
    ckpt = cfg.moge2_checkpoint if version == "v2" else cfg.moge3_checkpoint
    _log(f"MoGe-{version[1]} ({ckpt}) on {dev}")
    model = MoGeModel.from_pretrained(ckpt).to(dev).eval()
    H, W = image.shape[:2]
    x = torch.from_numpy(image.copy()).to(dev).float().permute(2, 0, 1) / 255.0
    with torch.inference_mode():
        out = model.infer(x, resolution_level=cfg.moge_resolution_level,
                          fov_x=cfg.fov_deg if cfg.fov_deg > 0 else None,
                          use_fp16=dev.type == "cuda")
    depth = out["depth"].float().cpu().numpy()
    valid = out["mask"].cpu().numpy() if "mask" in out else np.isfinite(depth)
    K = out["intrinsics"].float().cpu().numpy().astype(np.float64).copy()
    K[0] *= W  # MoGe returns normalised intrinsics
    K[1] *= H
    if not (np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0):
        raise RuntimeError(f"MoGe returned invalid intrinsics {K.tolist()}; wrong checkpoint?")
    normal = out["normal"].float().cpu().numpy() if "normal" in out else None
    depth, sky = _finalize(np.nan_to_num(depth, nan=0.0, posinf=0.0), valid, cfg)
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return DepthResult(depth, K, True, f"moge-{version}", normal, sky)


# ----------------------------------------------------------------------------- Depth Anything V2

def _dav2(image: np.ndarray, cfg: DepthConfig, fov: float) -> DepthResult:
    import cv2
    import onnxruntime as ort

    path = Path(cfg.dav2_onnx) if cfg.dav2_onnx and not cfg.dav2_onnx.startswith("http") else None
    if path is None:
        url = cfg.dav2_onnx or DAV2_URL
        path = download(url, url.rsplit("/", 1)[-1])
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(path), providers=providers)
    H, W = image.shape[:2]
    # Shorter side -> dav2_input_size, both sides multiples of 14 (as in the official transform).
    s = cfg.dav2_input_size / min(H, W)
    h, w = (max(14, int(round(H * s / 14)) * 14), max(14, int(round(W * s / 14)) * 14))
    x = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    _log(f"Depth Anything V2 ({path.name}) at {w}x{h}")
    disp = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    disp = cv2.resize(disp.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    valid = disp > 1e-3 * float(disp.max())  # before the offset: true "nothing there" (sky)
    disp = disp + cfg.dav2_disparity_offset * float(disp.max())
    depth = np.where(valid, 1.0 / np.maximum(disp, 1e-12), 0.0)
    depth *= cfg.median_depth / float(np.median(depth[valid]))
    depth, sky = _finalize(depth, valid, cfg)
    return DepthResult(depth, intrinsics_from_fov(W, H, fov), False, "depth-anything-v2", None, sky)


# ----------------------------------------------------------------------------- file

def _file(image: np.ndarray, cfg: DepthConfig, fov: float) -> DepthResult:
    import cv2

    p = Path(cfg.depth_file)
    if p.suffix == ".npy":
        depth = np.load(p).astype(np.float32)
    else:
        depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32) * cfg.depth_png_scale
    H, W = image.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    depth, sky = _finalize(depth, depth > 0, cfg)
    return DepthResult(depth, intrinsics_from_fov(W, H, fov), True, "file", None, sky)


def estimate_depth(image: np.ndarray, cfg: DepthConfig, exif_fov: Optional[float] = None,
                   device: str = "auto") -> DepthResult:
    """image: (H, W, 3) uint8 RGB."""
    fov = cfg.fov_deg if cfg.fov_deg > 0 else (exif_fov or cfg.default_fov_deg)
    if cfg.model == "moge2":
        res = _moge(image, cfg, "v2", device)
    elif cfg.model == "moge3":
        res = _moge(image, cfg, "v3", device)
    elif cfg.model == "dav2":
        _log(f"horizontal FoV {fov:.1f} deg ({'set' if cfg.fov_deg > 0 else 'EXIF' if exif_fov else 'default'})")
        res = _dav2(image, cfg, fov)
    elif cfg.model == "file":
        res = _file(image, cfg, fov)
    else:
        raise ValueError(f"Unknown depth model '{cfg.model}' (moge2, moge3, dav2, file)")
    fov_est = math.degrees(2 * math.atan(0.5 * image.shape[1] / res.K[0, 0]))
    _log(f"{res.model}: median depth {np.median(res.depth):.2f}{' m' if res.metric else ' (relative)'}, "
         f"FoV {fov_est:.1f} deg, far/sky pixels {100 * res.sky.mean():.1f}%")
    return res
