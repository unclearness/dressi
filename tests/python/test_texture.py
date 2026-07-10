"""texture: nearest sampling forward, tex gradient vs finite differences.

The gradient test uses a 1:1 screen<->texel mapping (full-screen quad,
identity UV, screen res == tex res) where the inverse-UV gather equals the
exact VJP; the general many-to-one case is the paper's approximation and
is exercised by the convergence test instead.
"""

import pytest
import torch

import dressi.torch as dr


def fullscreen_quad():
    # Covers all of NDC; uv spans [0,1]^2
    pos = torch.tensor(
        [
            [-1.0, -1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
    )
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tri = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
    return pos, uv, tri


def test_texture_nearest_forward(ctx):
    res = [8, 8]
    pos, uv, tri = fullscreen_quad()
    tex = torch.arange(8 * 8 * 1, dtype=torch.float32).reshape(8, 8, 1)

    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    uv_img, _ = dr.interpolate(uv, rast, tri)
    out = dr.texture(tex, uv_img)
    assert out.shape == (1, 8, 8, 1)

    # identity mapping: pixel (x, y) samples texel (x, y)
    for iy in range(res[0]):
        for ix in range(res[1]):
            assert out[0, iy, ix, 0].item() == pytest.approx(
                tex[iy, ix, 0].item(), abs=1e-4), (ix, iy)


def test_texture_grad_matches_fd(ctx):
    res = [8, 8]
    pos, uv, tri = fullscreen_quad()
    tex0 = torch.rand(8, 8, 2)
    weight = torch.rand(1, res[0], res[1], 2) - 0.5

    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    uv_img, _ = dr.interpolate(uv, rast, tri)

    def loss_of(t):
        return (dr.texture(t, uv_img) * weight).sum()

    tex = tex0.clone().requires_grad_(True)
    loss_of(tex).backward()
    analytic = tex.grad.clone()

    h = 1e-2  # linear in tex -> FD exact up to fp32 noise
    flat0 = tex0.reshape(-1)
    for i in range(0, flat0.numel(), 7):  # sample every 7th texel component
        d = torch.zeros_like(flat0)
        d[i] = h
        d = d.reshape(tex0.shape)
        with torch.no_grad():
            fd = (loss_of(tex0 + d) - loss_of(tex0 - d)) / (2 * h)
        assert analytic.reshape(-1)[i].item() == pytest.approx(
            fd.item(), abs=1e-2, rel=1e-3), f"texel component {i}"


def test_texture_infer_no_provenance_needed_for_grads_off(ctx):
    res = [8, 8]
    pos, uv, tri = fullscreen_quad()
    tex = torch.rand(4, 4, 3)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    uv_img, _ = dr.interpolate(uv, rast, tri)
    with torch.no_grad():
        out = dr.texture(tex, uv_img)
    assert out.shape == (1, 8, 8, 3)
