"""interpolate: forward vs manual barycentric math, backward vs finite
differences (the whole chain is linear in attr, so FD is exact)."""

import pytest
import torch

import dressi.torch as dr


def make_scene():
    pos = torch.tensor(
        [
            [-0.8, -0.8, 0.2, 1.0],
            [0.8, -0.6, 0.2, 1.0],
            [-0.6, 0.8, 0.2, 1.0],
            [0.9, 0.9, 0.2, 1.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)
    return pos, tri


def test_interpolate_forward_matches_bary(ctx):
    res = [16, 16]
    pos, tri = make_scene()
    attr = torch.tensor(
        [[1.0, 10.0, -1.0], [2.0, 20.0, -2.0], [3.0, 30.0, -3.0],
         [4.0, 40.0, -4.0]])

    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    out, _ = dr.interpolate(attr, rast, tri)
    assert out.shape == (1, res[0], res[1], 3)

    covered = 0
    for iy in range(res[0]):
        for ix in range(res[1]):
            u, v, _, tid = rast[0, iy, ix].tolist()
            if tid == 0.0:
                assert torch.count_nonzero(out[0, iy, ix]).item() == 0
                continue
            covered += 1
            i0, i1, i2 = tri[int(tid) - 1].tolist()
            expected = u * attr[i0] + v * attr[i1] + (1 - u - v) * attr[i2]
            got = out[0, iy, ix]
            assert torch.allclose(got, expected, atol=2e-3), (ix, iy, tid)
    assert covered > 20


def test_interpolate_grad_matches_fd(ctx):
    res = [12, 12]
    pos, tri = make_scene()
    attr0 = torch.tensor([[0.5], [1.5], [-0.7], [2.2]])
    weight = torch.linspace(-1.0, 1.0, res[0] * res[1]).reshape(
        1, res[0], res[1], 1)

    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)

    def loss_of(a):
        out, _ = dr.interpolate(a, rast, tri)
        return (out * weight).sum()

    attr = attr0.clone().requires_grad_(True)
    loss_of(attr).backward()
    analytic = attr.grad.clone()

    h = 1e-2  # linear in attr -> FD is exact up to fp32 noise
    for i in range(attr0.numel()):
        d = torch.zeros_like(attr0).reshape(-1)
        d[i] = h
        d = d.reshape(attr0.shape)
        with torch.no_grad():
            fd = (loss_of(attr0 + d) - loss_of(attr0 - d)) / (2 * h)
        assert analytic.reshape(-1)[i].item() == pytest.approx(
            fd.item(), abs=1e-2, rel=1e-3), f"vertex {i}"


def test_interpolate_batched_attr(ctx):
    res = [8, 8]
    pos, tri = make_scene()
    rast, _ = dr.rasterize(ctx, torch.stack([pos, pos]), tri, resolution=res)
    attr = torch.rand(2, 4, 2)
    out, _ = dr.interpolate(attr, rast, tri)
    assert out.shape == (2, 8, 8, 2)
    # same pos both items, different attr -> different outputs
    assert not torch.equal(out[0], out[1])
