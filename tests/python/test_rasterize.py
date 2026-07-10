"""rasterize value tests against analytic expectations.

The expected values mirror the engine's screen mapping
(cpu_raster.cpp ProjectToScreen): sx = (x/w*0.5+0.5)*W, row 0 <-> NDC
y=-1, pixel centers at +0.5 -- the same mapping nvdiffrast uses at the
tensor level.
"""

import numpy as np
import pytest
import torch

import dressi.torch as dr


def edge_barycentrics(pts, px, py):
    """Affine barycentrics (w0, w1, w2) of pixel center (px, py)."""
    (x0, y0), (x1, y1), (x2, y2) = pts
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    w0 = ((x1 - px) * (y2 - py) - (y1 - py) * (x2 - px)) / area
    w1 = ((x2 - px) * (y0 - py) - (y2 - py) * (x0 - px)) / area
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def screen_pts(pos, res):
    h, w = res
    pts = []
    for v in pos:
        sx = (v[0] / v[3] * 0.5 + 0.5) * w
        sy = (v[1] / v[3] * 0.5 + 0.5) * h
        pts.append((sx, sy))
    return pts


def test_single_triangle_barycentrics(ctx):
    res = (8, 8)
    pos = torch.tensor(
        [
            [-0.5, -0.5, 0.2, 1.0],
            [0.5, -0.5, 0.2, 1.0],
            [-0.5, 0.5, 0.2, 1.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    rast, rast_db = dr.rasterize(ctx, pos, tri, resolution=list(res))
    assert rast.shape == (1, res[0], res[1], 4)
    assert torch.count_nonzero(rast_db) == 0

    pts = screen_pts(pos.tolist(), res)
    covered = 0
    for iy in range(res[0]):
        for ix in range(res[1]):
            px, py = ix + 0.5, iy + 0.5
            w0, w1, w2 = edge_barycentrics(pts, px, py)
            inside = min(w0, w1, w2) > 1e-4  # skip exact-edge ambiguity
            outside = min(w0, w1, w2) < -1e-4
            got = rast[0, iy, ix]
            if inside:
                covered += 1
                assert got[3].item() == pytest.approx(1.0), (ix, iy)
                assert got[0].item() == pytest.approx(w0, abs=1e-3), (ix, iy)
                assert got[1].item() == pytest.approx(w1, abs=1e-3), (ix, iy)
                # constant z: z/w = 0.2 everywhere on the face
                assert got[2].item() == pytest.approx(0.2, abs=1e-4), (ix, iy)
            elif outside:
                assert got[3].item() == 0.0, (ix, iy)
                assert torch.count_nonzero(got).item() == 0, (ix, iy)
    assert covered > 0


def test_perspective_correct_barycentrics(ctx):
    # Unequal w: perspective-correct interpolation differs from affine
    res = (16, 16)
    pos = torch.tensor(
        [
            [-0.8, -0.8, 0.0, 1.0],
            [1.6, -1.6, 1.0, 2.0],
            [-1.6, 1.6, 1.0, 2.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=list(res))

    pts = screen_pts(pos.tolist(), res)
    ws = [1.0, 2.0, 2.0]
    checked = 0
    for iy in range(res[0]):
        for ix in range(res[1]):
            if rast[0, iy, ix, 3].item() == 0.0:
                continue
            px, py = ix + 0.5, iy + 0.5
            l0, l1, l2 = edge_barycentrics(pts, px, py)
            if min(l0, l1, l2) < 1e-3:
                continue  # stay clear of edges
            denom = l0 / ws[0] + l1 / ws[1] + l2 / ws[2]
            u = (l0 / ws[0]) / denom
            v = (l1 / ws[1]) / denom
            assert rast[0, iy, ix, 0].item() == pytest.approx(u, abs=2e-3)
            assert rast[0, iy, ix, 1].item() == pytest.approx(v, abs=2e-3)
            # z/w: perspective-correct z over w
            zw = (l0 * pos[0, 2] / ws[0] + l1 * pos[1, 2] / ws[1]
                  + l2 * pos[2, 2] / ws[2]) / denom / (
                      1.0 / denom)  # z_pc / w_pc
            # z_pc = sum(l_i z_i / w_i)/denom, w_pc = sum(l_i w_i / w_i)/denom
            w_pc = (l0 + l1 + l2) / denom  # since w_i/w_i = 1
            z_pc = (l0 * pos[0, 2].item() / ws[0] + l1 * pos[1, 2].item() / ws[1]
                    + l2 * pos[2, 2].item() / ws[2]) / denom
            assert rast[0, iy, ix, 2].item() == pytest.approx(
                z_pc / w_pc, abs=2e-3)
            checked += 1
    assert checked > 10


def test_depth_test_two_triangles(ctx):
    # Near triangle (z/w=0.25 -> vk depth 0.625... in GL terms z=-0.5..)
    # occludes the far one where they overlap; ids disambiguate
    res = (8, 8)
    pos = torch.tensor(
        [
            # far big triangle (z_gl = 0.5)
            [-0.9, -0.9, 0.5, 1.0],
            [0.9, -0.9, 0.5, 1.0],
            [0.0, 0.9, 0.5, 1.0],
            # near smaller triangle (z_gl = -0.5)
            [-0.4, -0.4, -0.5, 1.0],
            [0.4, -0.4, -0.5, 1.0],
            [0.0, 0.4, -0.5, 1.0],
        ]
    )
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=list(res))

    tid = rast[0, :, :, 3]
    assert (tid == 2.0).any(), "near triangle must be visible"
    assert (tid == 1.0).any(), "far triangle must be visible outside overlap"
    # center pixel is inside both -> near wins
    assert tid[3, 3].item() == 2.0
    assert rast[0, 3, 3, 2].item() == pytest.approx(-0.5, abs=1e-4)


def test_engine_cache_reuse(ctx):
    res = (6, 6)  # resolutions unique to this test (session-shared ctx)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    pos1 = torch.tensor(
        [[-0.5, -0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0], [-0.5, 0.5, 0.0, 1.0]])
    pos2 = pos1 * torch.tensor([1.5, 1.5, 1.0, 1.0])

    n_before = len(ctx._engines)
    r1, _ = dr.rasterize(ctx, pos1, tri, resolution=list(res))
    n_after_first = len(ctx._engines)
    r2, _ = dr.rasterize(ctx, pos2, tri, resolution=list(res))
    assert len(ctx._engines) == n_after_first  # same signature -> cache hit
    assert n_after_first == n_before + 1
    assert not torch.equal(r1, r2)  # new values did flow through

    # different resolution -> new engine
    dr.rasterize(ctx, pos1, tri, resolution=[12, 12])
    assert len(ctx._engines) == n_after_first + 1


def test_batch_loop(ctx):
    res = (4, 4)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    base = torch.tensor(
        [[-0.5, -0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0], [-0.5, 0.5, 0.0, 1.0]])
    pos = torch.stack([base, base * torch.tensor([0.2, 0.2, 1.0, 1.0])])
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=list(res))
    assert rast.shape == (2, 4, 4, 4)
    # smaller triangle covers fewer pixels
    assert (rast[1, :, :, 3] > 0).sum() < (rast[0, :, :, 3] > 0).sum()
