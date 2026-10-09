"""The fragment rasterizer must match a brute-force reference of the 3DGS image model."""

import math

import torch

from ssplat.rasterize import ALPHA_MAX, ALPHA_MIN, T_MIN, Camera, project, rasterize


def _random_scene(n=40, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g, dtype=dtype) * torch.tensor([2.0, 1.5, 2.0], dtype=dtype)
    means += torch.tensor([-1.0, -0.75, 2.0], dtype=dtype)
    quats = torch.randn(n, 4, generator=g, dtype=dtype)
    scales = torch.rand(n, 3, generator=g, dtype=dtype) * 0.08 + 0.01
    opac = torch.rand(n, generator=g, dtype=dtype) * 0.9 + 0.05
    colors = torch.rand(n, 3, generator=g, dtype=dtype)
    return means, quats, scales, opac, colors


def _camera(W=48, H=36, dtype=torch.float64):
    K = torch.tensor([[40.0, 0, W / 2 + 0.3], [0, 41.0, H / 2 - 0.7], [0, 0, 1]], dtype=dtype)
    return Camera(K, torch.eye(4, dtype=dtype), W, H)


def _brute_force(means, quats, scales, opac, colors, cam):
    """Per-pixel loop over all Gaussians, sorted by depth: the reference 3DGS rules."""
    proj = project(means, quats, scales, cam)
    order = torch.argsort(proj["depth"])
    H, W = cam.height, cam.width
    img = torch.zeros(H, W, 3, dtype=means.dtype)
    alpha_img = torch.zeros(H, W, dtype=means.dtype)
    for y in range(H):
        for x in range(W):
            T = 1.0
            for i in order.tolist():
                if proj["depth"][i] <= 1e-3:
                    continue
                dx = x + 0.5 - proj["mean2d"][i, 0]
                dy = y + 0.5 - proj["mean2d"][i, 1]
                if max(abs(dx), abs(dy)) > proj["radius"][i]:
                    continue
                a, b, c = proj["conic"][i]
                power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
                if power > 0:
                    continue
                alpha = min(ALPHA_MAX, float(opac[i] * math.exp(power)))
                if alpha < ALPHA_MIN:
                    continue
                if T * (1 - alpha) < T_MIN:
                    break
                img[y, x] += T * alpha * colors[i]
                alpha_img[y, x] += T * alpha
                T *= 1 - alpha
    return img, alpha_img


def test_matches_brute_force():
    means, quats, scales, opac, colors = _random_scene()
    cam = _camera()
    out = rasterize(means, quats, scales, opac, colors, cam)
    ref_img, ref_alpha = _brute_force(means, quats, scales, opac, colors, cam)
    assert out["n_fragments"] > 100
    assert torch.allclose(out["rgb"], ref_img, atol=1e-9)
    assert torch.allclose(out["alpha"], ref_alpha, atol=1e-9)


def test_gradients_match_finite_differences():
    means, quats, scales, opac, colors = _random_scene(n=12, seed=1)
    cam = _camera(24, 18)
    params = [t.clone().requires_grad_(True) for t in (means, scales, opac, colors)]

    def f(m, s, o, c):
        out = rasterize(m, quats, s, o, c, cam)
        return out["rgb"].sum() + 0.5 * out["depth"].sum()

    # The fragment set is a piecewise-constant function of the parameters; a tiny step keeps
    # it fixed, so the comparison is between the analytic and numerical gradient of one piece.
    assert torch.autograd.gradcheck(f, params, eps=1e-7, atol=1e-5, rtol=1e-4)


def test_background_and_empty_scene():
    cam = _camera(8, 6)
    bg = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    z = torch.zeros(0, 3, dtype=torch.float64)
    out = rasterize(z, torch.zeros(0, 4, dtype=torch.float64), z, torch.zeros(0, dtype=torch.float64), z,
                    cam, background=bg)
    assert torch.allclose(out["rgb"], bg.expand(6, 8, 3))
    assert out["alpha"].abs().max() == 0
