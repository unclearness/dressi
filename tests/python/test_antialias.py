"""antialias: forward behavior, color/pos gradients, and a mini silhouette
optimization (the nvdiffrast AA workflow end-to-end)."""

import pytest
import torch

import dressi.torch as dr


def tri_scene(dx=0.0, dy=0.0):
    pos = torch.tensor(
        [
            [-0.55 + dx, -0.45 + dy, 0.2, 1.0],
            [0.62 + dx, -0.38 + dy, 0.2, 1.0],
            [-0.35 + dx, 0.58 + dy, 0.2, 1.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return pos, tri


def render_mask(ctx, pos, tri, res, n_samples=0):
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    ones = torch.ones(pos.shape[0], 1)
    mask, _ = dr.interpolate(ones, rast, tri)
    return dr.antialias(mask, rast, pos, tri, n_samples=n_samples)


def test_antialias_forward_blends_boundary_only(ctx):
    res = [32, 32]
    pos, tri = tri_scene()
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    ones = torch.ones(3, 1)
    mask, _ = dr.interpolate(ones, rast, tri)
    aa = dr.antialias(mask, rast, pos, tri)

    tid = rast[0, :, :, 3]
    # pixels whose full 3x3 neighborhood shares the same id are untouched
    same = torch.ones_like(tid, dtype=torch.bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = torch.roll(tid, shifts=(dy, dx), dims=(0, 1))
            same &= shifted == tid
    interior = same.clone()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False

    diff = (aa[0, :, :, 0] - mask[0, :, :, 0]).abs()
    assert torch.all(diff[interior] < 1e-6)
    assert diff[~interior].max() > 1e-3  # boundary pixels actually blend


def test_antialias_color_grad_matches_fd(ctx):
    res = [16, 16]
    pos, tri = tri_scene()
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    ones0 = torch.full((3, 1), 0.8)
    weight = torch.rand(1, res[0], res[1], 1) - 0.5

    def loss_of(a):
        m, _ = dr.interpolate(a, rast, tri)
        return (dr.antialias(m, rast, pos, tri) * weight).sum()

    a = ones0.clone().requires_grad_(True)
    loss_of(a).backward()
    analytic = a.grad.clone()

    h = 1e-2  # AA is linear in the color image
    for i in range(3):
        d = torch.zeros_like(ones0)
        d[i, 0] = h
        with torch.no_grad():
            fd = (loss_of(ones0 + d) - loss_of(ones0 - d)) / (2 * h)
        assert analytic[i, 0].item() == pytest.approx(
            fd.item(), abs=2e-2, rel=2e-2), f"vertex {i}"


def test_antialias_pos_grad_direction(ctx):
    # Exact vertex backward (n_samples=0) vs central differences: the AA
    # forward keeps a residual discontinuity at coverage flips, so compare
    # by cosine similarity over the x/y components rather than pointwise
    res = [32, 32]
    pos0, tri = tri_scene()
    with torch.no_grad():
        target = render_mask(ctx, tri_scene(0.08, -0.05)[0], tri, res)

    def loss_of(p):
        return ((render_mask(ctx, p, tri, res) - target) ** 2).sum()

    p = pos0.clone().requires_grad_(True)
    loss_of(p).backward()
    analytic = p.grad[:, :2].reshape(-1)
    assert analytic.abs().max() > 0

    h = 2e-3
    fd = torch.zeros_like(analytic)
    k = 0
    for i in range(3):
        for c in range(2):
            d = torch.zeros_like(pos0)
            d[i, c] = h
            with torch.no_grad():
                fd[k] = (loss_of(pos0 + d) - loss_of(pos0 - d)) / (2 * h)
            k += 1
    cos = torch.dot(analytic, fd) / (analytic.norm() * fd.norm() + 1e-12)
    assert cos.item() > 0.9, f"cosine {cos.item()}: analytic {analytic}, fd {fd}"


def test_antialias_silhouette_optimization(ctx):
    torch.manual_seed(0)
    res = [32, 32]
    pos0, tri = tri_scene()
    with torch.no_grad():
        target = render_mask(ctx, tri_scene(0.15, -0.1)[0], tri, res)

    offset = torch.zeros(1, 2, requires_grad=True)
    opt = torch.optim.Adam([offset], lr=2e-2)

    def loss_fn():
        p = pos0.clone()
        p = torch.cat([pos0[:, :2] + offset, pos0[:, 2:]], dim=1)
        return ((render_mask(ctx, p, tri, res, n_samples=8) - target) ** 2).sum()

    loss_start = loss_fn().item()
    for _ in range(150):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    loss_end = loss_fn().item()
    assert loss_end < loss_start * 0.1, f"{loss_start} -> {loss_end}"
    assert offset.detach()[0, 0].item() == pytest.approx(0.15, abs=0.03)
    assert offset.detach()[0, 1].item() == pytest.approx(-0.1, abs=0.03)
