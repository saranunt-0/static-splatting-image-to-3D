"""Image inpainting for the hidden layer: LaMa (default) or OpenCV Telea (no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"


class TeleaInpainter:
    name = "telea"

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.inpaint(image, mask.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)


class LamaInpainter:
    """big-lama (Suvorov et al., WACV 2022; Apache-2.0) as TorchScript.

    Fully convolutional with Fourier convolutions: good at continuing large
    regular structures such as walls, floors, tiles and ceilings.
    """

    name = "lama"

    def __init__(self, checkpoint: str = "", device: str = "auto"):
        import hashlib

        import torch

        from .depth import _device, download

        path = Path(checkpoint) if checkpoint else download(LAMA_URL, "big-lama.pt")
        if not checkpoint:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != LAMA_SHA256:
                raise RuntimeError(f"Checksum mismatch for {path}; delete it and retry.")
        self.device = _device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image (H, W, 3) uint8, mask (H, W) bool (True = fill). Returns uint8."""
        import torch

        if not mask.any():
            return image.copy()
        H, W = mask.shape
        ph, pw = (-H) % 8, (-W) % 8
        img = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        m = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
        x = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        mt = torch.from_numpy(m.astype(np.float32)).to(self.device)[None, None]
        with torch.inference_mode():
            out = self.model(x, mt)
        out = (out[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()[:H, :W]
        return np.where(mask[..., None], out, image)


def make_inpainter(kind: str, checkpoint: str = "", device: str = "auto"):
    if kind == "lama":
        return LamaInpainter(checkpoint, device)
    if kind == "telea":
        return TeleaInpainter()
    raise ValueError(f"Unknown inpainter '{kind}' (lama, telea)")
