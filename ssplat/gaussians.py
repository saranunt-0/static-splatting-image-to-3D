"""Gaussian set container (numpy) and conversions to torch tensors and .ply splats."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Dict

import numpy as np

from .splats import Splats

if TYPE_CHECKING:  # pragma: no cover
    import torch

SH_C0 = 0.28209479177387814


@dataclass
class GaussianSet:
    """Activated parameters: colours in [0, 1] (flat, SH band 0), opacity in [0, 1], sigma scales."""

    means: np.ndarray       # (N, 3) float32, world = input camera frame
    quats: np.ndarray       # (N, 4) wxyz
    scales: np.ndarray      # (N, 3) sigma
    opacities: np.ndarray   # (N,)
    colors: np.ndarray      # (N, 3)
    layer: np.ndarray = field(default=None)  # (N,) int8: 0 visible surface, 1 hidden / border layer

    def __post_init__(self):
        if self.layer is None:
            self.layer = np.zeros(len(self.means), np.int8)

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def subset(self, mask: np.ndarray) -> "GaussianSet":
        return GaussianSet(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    @staticmethod
    def concat(*sets: "GaussianSet") -> "GaussianSet":
        return GaussianSet(**{f.name: np.concatenate([getattr(s, f.name) for s in sets])
                              for f in fields(GaussianSet)})

    def to_torch(self, device="cpu") -> Dict[str, "torch.Tensor"]:
        import torch

        return {f.name: torch.as_tensor(np.ascontiguousarray(getattr(self, f.name)), device=device)
                for f in fields(self)}

    @classmethod
    def from_torch(cls, d) -> "GaussianSet":
        return cls(**{f.name: d[f.name].detach().cpu().numpy() for f in fields(cls)})

    def to_splats(self) -> Splats:
        op = np.clip(self.opacities.astype(np.float64), 1e-6, 1 - 1e-6)
        return Splats(
            means=self.means.astype(np.float32),
            sh0=((self.colors - 0.5) / SH_C0).astype(np.float32),
            shN=np.zeros((len(self), 0, 3), np.float32),
            opacities=np.log(op / (1 - op)).astype(np.float32),
            scales=np.log(np.maximum(self.scales, 1e-12)).astype(np.float32),
            quats=self.quats.astype(np.float32),
        )

    @classmethod
    def from_splats(cls, s: Splats) -> "GaussianSet":
        return cls(
            means=s.means.astype(np.float32),
            quats=s.quats.astype(np.float32),
            scales=np.exp(s.scales).astype(np.float32),
            opacities=(1 / (1 + np.exp(-s.opacities))).astype(np.float32),
            colors=np.clip(0.5 + SH_C0 * s.sh0, 0, None).astype(np.float32),
        )
