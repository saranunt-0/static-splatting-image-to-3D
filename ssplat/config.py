"""Pipeline configuration: one dataclass per stage plus quality presets.

The same config drives the CLI (``python -m ssplat.run``) and the Colab
notebook, and is saved next to the outputs so every result is reproducible.
Override any field from the CLI with ``--set section.field=value``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DepthConfig:
    """Monocular geometry (see ssplat/depth.py and docs/RESEARCH.md)."""

    # moge2  : MoGe-2 (Microsoft, NeurIPS 2025, MIT). Metric point map + camera FoV + normals.
    #          Best accuracy among permissively licensed single-image models. Default.
    # moge3  : MoGe-3 (2026). Sharper detail, needs a CUDA GPU (sparse-conv refinement).
    # dav2   : Depth Anything V2 (relative disparity, ONNX, CPU friendly). Has no FoV and no
    #          metric scale: uses fov_deg / EXIF / default_fov_deg and median_depth instead.
    # file   : your own depth map (.npy in metres, or 16-bit .png with depth_png_scale).
    model: str = "moge2"
    moge2_checkpoint: str = "Ruicheng/moge-2-vitl-normal"
    moge3_checkpoint: str = "Ruicheng/moge-3-vitl"
    # MoGe resolution level 0..9 (higher = more ViT tokens = finer, slower).
    moge_resolution_level: int = 9
    # Depth Anything V2: ONNX file (path or URL). Empty = Large model from the
    # fabio-sim/Depth-Anything-ONNX GitHub release (Apache-2.0 code; ViT-L weights CC-BY-NC-4.0).
    dav2_onnx: str = ""
    # Shorter image side fed to Depth Anything (multiple of 14).
    dav2_input_size: int = 518
    # Shift of the affine-invariant disparity as a fraction of its maximum (dav2 only).
    # 0 assumes the predicted disparity is proportional to 1/depth.
    dav2_disparity_shift: float = 0.0
    # Horizontal field of view in degrees. 0 = from the model (MoGe), else from EXIF,
    # else default_fov_deg.
    fov_deg: float = 0.0
    default_fov_deg: float = 60.0
    # Relative-depth models only: depth (in metres) assigned to the median pixel.
    median_depth: float = 3.0
    depth_file: str = ""
    depth_png_scale: float = 0.001
    # Far depths are clamped to this multiple of the median depth (sky, windows).
    max_depth_ratio: float = 25.0


@dataclass
class LiftConfig:
    """Depth map -> one Gaussian per pixel (see ssplat/lift.py)."""

    # Working resolution: the image is resized so its longest side is at most this.
    # One Gaussian per pixel, so this sets the Gaussian count (1024 x 768 = 786k).
    max_long_edge: int = 1024
    # Gaussian footprint standard deviation in pixels. Together with the renderer's
    # 0.3 px^2 low-pass filter this tiles the image without gaps.
    footprint_sigma: float = 0.45
    # Surface-normal thickness relative to the in-plane footprint (thin "surfels").
    thickness: float = 0.1
    # Snap depth values between foreground and background ("flying pixels") to one side.
    sharpen_edges: bool = True
    # Initial opacity of surface Gaussians.
    opacity: float = 0.98


@dataclass
class OcclusionConfig:
    """Hidden background layer + border extension (see ssplat/occlusion.py)."""

    enabled: bool = True
    # How far the viewer is expected to move, as a fraction of the median scene depth.
    # Sets how wide the hidden layer behind each depth edge and the border extension are.
    max_baseline: float = 0.08
    # A depth discontinuity is an occlusion edge if moving by max_baseline would open a
    # gap of at least this many pixels.
    edge_parallax_px: float = 1.0
    # ...and if the far side is at least this much (relative) farther away.
    edge_min_ratio: float = 0.03
    # lama: LaMa (Apache-2.0, big-lama TorchScript, CPU OK). telea: OpenCV, no model.
    inpainter: str = "lama"
    lama_checkpoint: str = ""
    # Extend the scene beyond the photo frame so moving the camera does not show black borders.
    extend_border: bool = True
    # Cap on the border extension and on the hidden layer width (fraction of the image width).
    max_extent_frac: float = 0.15
    # Extra pixels added to the hidden-layer band.
    band_margin_px: int = 3


@dataclass
class OptimConfig:
    """Single-view refinement of the Gaussians (see ssplat/optimize.py)."""

    # 0 disables the refinement (lifted Gaussians are exported as they are).
    steps: int = 300
    ssim_weight: float = 0.2
    # Keeps rendered depth on the monocular depth (the only geometric evidence we have).
    depth_weight: float = 0.5
    # Keeps every Gaussian close to its initial distance along its viewing ray.
    anchor_weight: float = 1.0
    # Keeps Gaussians flat (surface-like) so they look right from other viewpoints.
    flatten_weight: float = 0.01
    lr_means: float = 2e-5    # x median depth
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 2e-2
    lr_color: float = 5e-3
    # Optimise at a lower resolution than the Gaussians (1 = full). Speeds up CPU runs.
    render_scale: float = 1.0
    # auto: cuda if available, else mps, else cpu.
    device: str = "auto"
    seed: int = 0


@dataclass
class RenderConfig:
    """Preview videos and novel-view images (see ssplat/render.py)."""

    video: bool = True
    # swing: left-right; circle: small circle around the photo viewpoint; dolly: walk forward.
    trajectories: List[str] = field(default_factory=lambda: ["swing", "circle", "dolly"])
    frames: int = 90
    fps: int = 30
    # Camera motion amplitude as a fraction of the median depth (keep <= occlusion.max_baseline).
    motion: float = 0.06
    # Forward travel of the dolly trajectory as a fraction of the median depth.
    dolly: float = 0.25
    # Output resolution relative to the working resolution.
    scale: float = 1.0


@dataclass
class ExportConfig:
    """Files for 3D software and viewers (see ssplat/export.py)."""

    # Axis convention of the exported scene. The photo's camera sits at the origin.
    #   3dgs : OpenCV / COLMAP convention (x right, -y up, z forward); de-facto .ply standard.
    #   z_up : +Z up (Blender, Unreal).    y_up : +Y up (three.js, Unity, glTF).
    orientation: str = "3dgs"
    # Drop nearly transparent Gaussians (activated opacity below this).
    min_opacity: float = 0.02
    # ply is always written; spz / glb / html need Node.js (PlayCanvas splat-transform).
    formats: List[str] = field(default_factory=lambda: ["ply", "html"])


@dataclass
class PipelineConfig:
    # layered: depth -> Gaussians + hidden layer -> refinement (this repository's method).
    # sharp  : Apple SHARP feed-forward model (research-only licence; optional install).
    method: str = "layered"
    # SHARP checkpoint (.pt). Empty = download Apple's release on first use.
    sharp_checkpoint: str = ""
    depth: DepthConfig = field(default_factory=DepthConfig)
    lift: LiftConfig = field(default_factory=LiftConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        cfg = cls()
        for section, values in data.items():
            if not isinstance(values, dict):
                if not hasattr(cfg, section):
                    raise KeyError(f"Unknown config option {section}")
                setattr(cfg, section, values)
                continue
            target = getattr(cfg, section)
            for key, value in values.items():
                if not hasattr(target, key):
                    raise KeyError(f"Unknown config option {section}.{key}")
                setattr(target, key, value)
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


PRESETS: Dict[str, Dict[str, Any]] = {
    # CPU-friendly: ~200k Gaussians, short refinement.
    "fast": {"lift": {"max_long_edge": 512}, "optim": {"steps": 150},
             "render": {"frames": 60}},
    "balanced": {"lift": {"max_long_edge": 768}, "optim": {"steps": 300}},
    # Default on a GPU.
    "quality": {"lift": {"max_long_edge": 1024}, "optim": {"steps": 500}},
}


def from_preset(name: str) -> PipelineConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Choose from {sorted(PRESETS)}")
    return PipelineConfig.from_dict(PRESETS[name])
