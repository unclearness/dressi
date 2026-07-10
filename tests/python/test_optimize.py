"""End-to-end: fit a texture through rasterize -> interpolate -> texture
with a plain torch Adam loop (the nvdiffrast texture-fitting workflow)."""

import torch

import dressi.torch as dr


def test_texture_fit_converges(ctx):
    torch.manual_seed(0)
    res = [24, 24]
    # Quad covering ~80% of the screen: texels map to several pixels each
    s = 0.8
    pos = torch.tensor(
        [
            [-s, -s, 0.0, 1.0],
            [s, -s, 0.0, 1.0],
            [-s, s, 0.0, 1.0],
            [s, s, 0.0, 1.0],
        ]
    )
    uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tri = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.int32)

    gt_tex = torch.rand(8, 8, 3)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=res)
    uv_img, _ = dr.interpolate(uv, rast, tri)
    with torch.no_grad():
        target = dr.texture(gt_tex, uv_img)

    tex = torch.full((8, 8, 3), 0.5, requires_grad=True)
    opt = torch.optim.Adam([tex], lr=5e-2)

    def loss_fn():
        pred = dr.texture(tex, uv_img)
        return ((pred - target) ** 2).mean()

    loss0 = loss_fn().item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    loss1 = loss_fn().item()

    assert loss1 < loss0 * 0.05, f"did not converge: {loss0} -> {loss1}"
