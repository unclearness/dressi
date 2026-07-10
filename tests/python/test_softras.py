"""rasterize_soft (HardSoftRas): channel semantics, depth peeling, pos
gradients, and a mini silhouette optimization through the sigmoid band."""

import pytest
import torch

import dressi.torch as dr


def tri_scene(dx=0.0, dy=0.0):
    pos = torch.tensor(
        [
            [-0.5 + dx, -0.42 + dy, 0.2, 1.0],
            [0.58 + dx, -0.35 + dy, 0.2, 1.0],
            [-0.32 + dx, 0.55 + dy, 0.2, 1.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return pos, tri


def soft_mask(ctx, pos, tri, res, radius=3.0, sigma=1.0):
    rs = dr.rasterize_soft(ctx, pos, tri, res, radius_px=radius)
    dist, cov = rs[..., 0], rs[..., 3]
    return cov * torch.sigmoid(dist / sigma)


def test_rasterize_soft_channels(ctx):
    res = [32, 32]
    pos, tri = tri_scene()
    rs = dr.rasterize_soft(ctx, pos, tri, res, radius_px=3.0)
    assert rs.shape == (1, 1, 32, 32, 4)
    dist = rs[0, 0, :, :, 0]
    fid = rs[0, 0, :, :, 1]
    cov = rs[0, 0, :, :, 3]

    hard_inside = (cov > 0.5) & (dist > 0.5)
    rim = (cov > 0.5) & (dist < -0.5)
    background = cov < 0.5
    assert hard_inside.sum() > 20
    assert rim.sum() > 10  # enlarged triangles produce a soft rim
    assert background.sum() > 200
    assert torch.all(fid[cov > 0.5] == 0.0)  # single face -> id 0
    # corners far from the triangle stay exactly zero
    assert torch.count_nonzero(rs[0, 0, 0, -1]) == 0


def test_rasterize_soft_two_layers(ctx):
    res = [24, 24]
    # Two stacked triangles; peel 1 exposes the far one under the near one
    pos = torch.tensor(
        [
            [-0.7, -0.7, 0.2, 1.0],
            [0.7, -0.7, 0.2, 1.0],
            [0.0, 0.7, 0.2, 1.0],
            [-0.5, -0.5, 0.6, 1.0],
            [0.5, -0.5, 0.6, 1.0],
            [0.0, 0.5, 0.6, 1.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    rs = dr.rasterize_soft(ctx, pos, tri, res, radius_px=2.0, peels=2)
    assert rs.shape == (1, 2, 24, 24, 4)
    fid0 = rs[0, 0, :, :, 1]
    cov0 = rs[0, 0, :, :, 3]
    fid1 = rs[0, 1, :, :, 1]
    cov1 = rs[0, 1, :, :, 3]
    # front layer is face 0 where both overlap; the second peel exposes
    # face 1 somewhere in the overlap
    center = (cov0 > 0.5) & (fid0 == 0.0)
    assert center.sum() > 10
    exposed = (cov1 > 0.5) & (fid1 == 1.0)
    assert exposed.sum() > 5


def test_rasterize_soft_grad_direction(ctx):
    res = [32, 32]
    pos0, tri = tri_scene()
    with torch.no_grad():
        target = soft_mask(ctx, tri_scene(0.06, -0.04)[0], tri, res)

    def loss_of(p):
        return ((soft_mask(ctx, p, tri, res) - target) ** 2).sum()

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
    assert cos.item() > 0.9, f"cosine {cos.item()}"


def test_rasterize_soft_silhouette_optimization(ctx):
    res = [32, 32]
    pos0, tri = tri_scene()
    with torch.no_grad():
        target = soft_mask(ctx, tri_scene(0.12, -0.08)[0], tri, res)

    offset = torch.zeros(1, 2, requires_grad=True)
    opt = torch.optim.Adam([offset], lr=2e-2)

    def loss_fn():
        p = torch.cat([pos0[:, :2] + offset, pos0[:, 2:]], dim=1)
        return ((soft_mask(ctx, p, tri, res) - target) ** 2).sum()

    loss_start = loss_fn().item()
    for _ in range(120):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    loss_end = loss_fn().item()
    assert loss_end < loss_start * 0.1, f"{loss_start} -> {loss_end}"
    assert offset.detach()[0, 0].item() == pytest.approx(0.12, abs=0.03)
    assert offset.detach()[0, 1].item() == pytest.approx(-0.08, abs=0.03)
