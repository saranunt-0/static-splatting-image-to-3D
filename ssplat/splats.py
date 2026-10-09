"""Read / write / transform standard 3DGS .ply files (numpy only, no GPU).

Layout follows the reference 3DGS implementation (and gsplat's exporter):
x y z [nx ny nz] f_dc_0..2 f_rest_0..(3K-1) opacity scale_0..2 rot_0..3, with
f_rest stored channel-major (all red coefficients, then green, then blue),
opacity as a logit, scales as log(sigma) and rotations as (w, x, y, z).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Real spherical-harmonic basis exactly as evaluated by 3DGS renderers.
_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
       -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(dirs: np.ndarray, degree: int) -> np.ndarray:
    """(M, 3) unit directions -> (M, (degree+1)^2) basis values, 3DGS sign convention."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    out = [np.full_like(x, _C0)]
    if degree >= 1:
        out += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [_C2[0] * xy, _C2[1] * yz, _C2[2] * (2 * zz - xx - yy), _C2[3] * xz, _C2[4] * (xx - yy)]
    if degree >= 3:
        out += [
            _C3[0] * y * (3 * xx - yy),
            _C3[1] * xy * z,
            _C3[2] * y * (4 * zz - xx - yy),
            _C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            _C3[4] * x * (4 * zz - xx - yy),
            _C3[5] * z * (xx - yy),
            _C3[6] * x * (xx - 3 * yy),
        ]
    return np.stack(out, axis=1)


def sh_rotation_matrices(R: np.ndarray, degree: int, num_samples: int = 256, seed: int = 0):
    """Per-band matrices X_l with coeffs' = X_l @ coeffs for a world rotation R.

    Rotating the scene by R must satisfy colour'(R d) = colour(d). Each band l is
    closed under rotation, so X_l is found exactly by least squares on sample
    directions (Y(R^T d) = M Y(d), and coeffs' = M^T coeffs).
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(num_samples, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    A = sh_basis(d, degree)
    B = sh_basis(d @ R, degree)  # rows are Y(R^T d_s)
    mats = []
    for l in range(1, degree + 1):
        sl = slice(l * l, (l + 1) * (l + 1))
        X, *_ = np.linalg.lstsq(A[:, sl], B[:, sl], rcond=None)  # A X = B  =>  X = M^T
        mats.append(X)
    return mats


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions (w, x, y, z), any norm -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Single 3x3 rotation matrix -> quaternion (w, x, y, z)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b; a is (4,), b is (N, 4); both (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


@dataclass
class Splats:
    means: np.ndarray      # (N, 3)
    sh0: np.ndarray        # (N, 3)     DC coefficients
    shN: np.ndarray        # (N, K, 3)  higher-order coefficients, K = (deg+1)^2 - 1
    opacities: np.ndarray  # (N,)       logits
    scales: np.ndarray     # (N, 3)     log sigma
    quats: np.ndarray      # (N, 4)     (w, x, y, z)

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.shN.shape[1] + 1))) - 1

    def subset(self, mask: np.ndarray) -> "Splats":
        return Splats(self.means[mask], self.sh0[mask], self.shN[mask],
                      self.opacities[mask], self.scales[mask], self.quats[mask])

    def finite_mask(self) -> np.ndarray:
        parts = [self.means, self.sh0, self.shN.reshape(len(self), -1), self.opacities[:, None],
                 self.scales, self.quats]
        ok = np.all(np.concatenate([np.isfinite(p) for p in parts], axis=1), axis=1)
        return ok & (np.linalg.norm(self.quats, axis=1) > 1e-8)

    def with_sh_degree(self, degree: int) -> "Splats":
        k = (degree + 1) ** 2 - 1
        if k >= self.shN.shape[1]:
            return self
        return Splats(self.means, self.sh0, self.shN[:, :k], self.opacities, self.scales, self.quats)

    def transformed(self, R: np.ndarray, t: np.ndarray | None = None, scale: float = 1.0) -> "Splats":
        """Apply x' = scale * R x + t (R a proper rotation) to positions, orientations and SH."""
        R = np.asarray(R, dtype=np.float64)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0:
            raise ValueError("R must be a proper rotation matrix")
        t = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64)
        means = (scale * self.means.astype(np.float64) @ R.T + t).astype(np.float32)
        quats = quat_multiply(matrix_to_quat(R), self.quats.astype(np.float64)).astype(np.float32)
        scales = (self.scales + np.log(scale)).astype(np.float32)
        shN = self.shN.astype(np.float64).copy()
        for l, X in enumerate(sh_rotation_matrices(R, self.sh_degree), start=1):
            sl = slice(l * l - 1, (l + 1) * (l + 1) - 1)  # shN excludes the DC term
            shN[:, sl, :] = np.einsum("ij,njc->nic", X, shN[:, sl, :])
        return Splats(means, self.sh0.copy(), shN.astype(np.float32), self.opacities.copy(), scales, quats)


def read_ply(path: str | Path) -> Splats:
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    names = v.dtype.names
    col = lambda n: np.asarray(v[n], dtype=np.float32)
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
    n = len(v)
    k = len(rest) // 3
    shN = (np.stack([col(r) for r in rest], axis=1).reshape(n, 3, k).transpose(0, 2, 1)
           if k else np.zeros((n, 0, 3), np.float32))
    return Splats(
        means=np.stack([col("x"), col("y"), col("z")], 1),
        sh0=np.stack([col(f"f_dc_{i}") for i in range(3)], 1),
        shN=shN,
        opacities=col("opacity"),
        scales=np.stack([col(f"scale_{i}") for i in range(3)], 1),
        quats=np.stack([col(f"rot_{i}") for i in range(4)], 1),
    )


def write_ply(splats: Splats, path: str | Path, with_normals: bool = True) -> Path:
    """Binary little-endian PLY in the reference 3DGS layout (read by virtually every tool)."""
    n, k = len(splats), splats.shN.shape[1]
    names = ["x", "y", "z"] + (["nx", "ny", "nz"] if with_normals else [])
    names += [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    cols = [splats.means]
    if with_normals:
        cols.append(np.zeros((n, 3), np.float32))
    cols += [splats.sh0, splats.shN.transpose(0, 2, 1).reshape(n, 3 * k),
             splats.opacities[:, None], splats.scales, splats.quats]
    data = np.ascontiguousarray(np.concatenate(cols, axis=1).astype("<f4"))
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n" + "".join(f"property float {m}\n" for m in names) + "end_header\n"
    path = Path(path)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    return path


def eval_color(splats: Splats, idx: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """RGB (before the +0.5 offset/clamp) of splats[idx] seen along unit `dirs` (M, 3)."""
    basis = sh_basis(dirs, splats.sh_degree)  # (M, B)
    coeffs = np.concatenate([splats.sh0[idx][:, None, :], splats.shN[idx]], axis=1)  # (M, B, 3)
    return np.einsum("mb,mbc->mc", basis, coeffs)
