"""Intrinsics, EXIF focal length and novel-view camera trajectories.

World frame = the photo's camera frame (OpenCV: x right, y down, z forward),
so the input view is the identity pose.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np


def intrinsics_from_fov(width: int, height: int, fov_x_deg: float) -> np.ndarray:
    fx = 0.5 * width / math.tan(math.radians(fov_x_deg) / 2)
    return np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=np.float64)


def fov_from_K(K: np.ndarray, width: int) -> float:
    return math.degrees(2 * math.atan(0.5 * width / K[0, 0]))


def fov_from_exif(path: str | Path) -> Optional[float]:
    """Horizontal FoV from the 35 mm-equivalent focal length in EXIF, if present.

    The 35 mm equivalent is defined on the diagonal (43.27 mm), which makes it
    independent of the image aspect ratio.
    """
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            w, h = im.size
            sub = exif.get_ifd(0x8769)  # Exif sub-IFD holds the focal length tags
            f35 = sub.get(0xA405) or exif.get(0xA405)  # FocalLengthIn35mmFilm
            orientation = exif.get(0x0112, 1)
    except Exception:
        return None
    if not f35 or float(f35) < 5:
        return None
    if orientation in (5, 6, 7, 8):  # stored rotated; width/height swap after transpose
        w, h = h, w
    diag_px = math.hypot(w, h)
    f_px = float(f35) * diag_px / 43.2666
    return math.degrees(2 * math.atan(0.5 * w / f_px))


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, -1.0, 0.0])) -> np.ndarray:
    """World->camera matrix (OpenCV axes) of a camera at `eye` looking at `target`."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: camera axes in world coordinates
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def translation_view(t: np.ndarray) -> np.ndarray:
    """Camera translated by t (world units) without rotation."""
    V = np.eye(4)
    V[:3, 3] = -np.asarray(t, dtype=np.float64)
    return V


def trajectory(kind: str, n: int, median_depth: float, motion: float, dolly: float) -> List[np.ndarray]:
    """Looping camera paths around the photo's viewpoint.

    swing : sideways sweep, looking at the scene centre (largest disocclusions).
    circle: circle of radius `motion` around the viewpoint, looking at the scene centre.
    dolly : straight forward into the scene and back (walking down a corridor).
    """
    r = motion * median_depth
    target = np.array([0.0, 0.0, median_depth])
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    views = []
    for a in s:
        if kind == "swing":
            eye = np.array([r * math.sin(a), 0.0, 0.0])
            views.append(look_at(eye, target))
        elif kind == "circle":
            eye = np.array([r * math.cos(a), r * math.sin(a), 0.0])
            views.append(look_at(eye, target))
        elif kind == "dolly":
            d = dolly * median_depth * 0.5 * (1 - math.cos(a))
            views.append(translation_view(np.array([0.0, 0.0, d])))
        else:
            raise ValueError(f"Unknown trajectory '{kind}' (swing, circle, dolly)")
    return views
