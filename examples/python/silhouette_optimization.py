"""Multi-view silhouette optimization (Python port of
examples/silhouette_optimization): deform an icosphere into the bunny from
its silhouettes.

Everything outside the rasterizer API is plain torch: cameras, mask loss,
regularizers, Adam. Following the C++ example, the uniform-Laplacian and
normal-consistency regularizers are combined at the GRADIENT level, scaled
relative to the data-gradient RMS — so the C++ example's tuned weights
(hardsoftras 0.15/0.4, aa 0.5/0.5) apply unchanged. The rasterizer API is
nvdiffrast drop-in, so the same code runs on either backend:

    python examples/python/silhouette_optimization.py                      # dressi (Vulkan)
    python examples/python/silhouette_optimization.py --backend nvdiffrast # CUDA
    python examples/python/silhouette_optimization.py --technique hardsoftras

The `aa` technique (rasterize + antialias on the mask) works on both
backends; `hardsoftras` (dr.rasterize_soft) is a dressi extension.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import torch

from common import (
    add_common_args,
    build_edges,
    build_face_pairs,
    circle_mvps,
    default_device,
    generate_icosphere,
    load_obj,
    resolve_backend,
    save_image,
    save_obj,
    tile_images,
    transform_clip,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--data", default="data/bunny", help="target mesh dir")
    ap.add_argument("--technique", default="aa",
                    choices=["aa", "hardsoftras"])
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--level", type=int, default=3, help="icosphere level")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--laplacian", type=float, default=None,
                    help="relative to the data-grad RMS "
                         "(default: technique-specific, as in C++)")
    ap.add_argument("--normal", type=float, default=None)
    ap.add_argument("--radius", type=float, default=3.0,
                    help="hardsoftras enlargement radius (px)")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="hardsoftras sigmoid bandwidth (px)")
    ap.add_argument("--peels", type=int, default=1)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    args.iters = args.iters or 300
    res = args.res or 128
    hardsoft = args.technique == "hardsoftras"
    # Tuned for this port (torch Adam + energy-derived normal gradient
    # differ subtly from the C++ in-graph optimizer, so the C++ weights
    # do not transfer 1:1)
    if args.laplacian is None:
        args.laplacian = 0.15 if hardsoft else 0.2
    if args.normal is None:
        args.normal = 0.4 if hardsoft else 0.2
    device = torch.device(args.device) if args.device else default_device()
    dr, ctx = resolve_backend(args.backend)
    if hardsoft and args.backend != "dressi":
        raise SystemExit("--technique hardsoftras requires --backend dressi")
    out_dir = pathlib.Path(args.out or f"silopt_py_out_{args.technique}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bunny_pos, _, bunny_tri = load_obj(
        pathlib.Path(args.data) / "bunny.obj")
    sphere_pos, sphere_tri = generate_icosphere(args.level)
    bunny_pos, bunny_tri = bunny_pos.to(device), bunny_tri.to(device)
    sphere_tri = sphere_tri.to(device)
    print(f"target: {len(bunny_pos)}v/{len(bunny_tri)}f  "
          f"sphere: {len(sphere_pos)}v/{len(sphere_tri)}f  "
          f"backend={args.backend} technique={args.technique} "
          f"device={device.type} {res}x{res} x{args.views} views")

    mvps = circle_mvps(args.views, device=device)
    edges = build_edges(sphere_tri).to(device)
    pairs = build_face_pairs(sphere_tri).to(device)

    def hard_mask(pos3, tri, mvp):
        clip = transform_clip(pos3, mvp)[None]
        rast, _ = dr.rasterize(ctx, clip, tri, resolution=[res, res])
        return (rast[..., 3:] > 0).float()[0]

    def render_aa(pos3, tri, mvp):
        clip = transform_clip(pos3, mvp)[None]
        rast, _ = dr.rasterize(ctx, clip, tri, resolution=[res, res])
        mask = (rast[..., 3:] > 0).float()
        return dr.antialias(mask, rast, clip, tri)[0]

    def render_hsr(pos3, tri, mvp):
        clip = transform_clip(pos3, mvp)
        rs = dr.rasterize_soft(ctx, clip, tri, [res, res],
                               radius_px=args.radius, peels=args.peels)[0]
        dist, cov = rs[..., 0], rs[..., 3]
        d_soft = cov * torch.sigmoid(dist / args.sigma)  # [K,H,W]
        soft = 1.0 - torch.prod(1.0 - d_soft, dim=0)  # Eq.6 over peels
        # Eq.5 split: hard coverage carries the value, the sigmoid band
        # carries the gradients (pure sigmoid leaves interior seams at K=1)
        hard = (cov[0] * (dist[0] >= 0).float()).detach()
        return torch.maximum(hard, soft)[..., None]

    render = render_hsr if hardsoft else render_aa

    # Targets, as in C++: the aa technique compares against AA'd masks (so
    # pred can match exactly); hardsoftras against hard masks
    with torch.no_grad():
        if hardsoft:
            targets = [hard_mask(bunny_pos, bunny_tri, m) for m in mvps]
        else:
            targets = [render_aa(bunny_pos, bunny_tri, m) for m in mvps]
    save_image(out_dir / "targets.png", tile_images(targets, 4))

    # C++ example: unit icosphere scaled to radius 0.55
    pos = (sphere_pos * 0.55).to(device).requires_grad_(True)
    opt = torch.optim.Adam([pos], lr=args.lr)

    def data_loss():
        return sum(((render(pos, sphere_tri, m) - t) ** 2).mean()
                   for m, t in zip(mvps, targets))

    def rms(g):
        return g.pow(2).mean().sqrt() + 1e-12

    def nbr_mean(p):
        n = torch.zeros_like(p)
        cnt = torch.zeros(p.shape[0], 1, dtype=p.dtype, device=p.device)
        one = torch.ones(edges.shape[0], 1, dtype=p.dtype, device=p.device)
        for a, b in ((0, 1), (1, 0)):
            n = n.index_add(0, edges[:, a], p[edges[:, b]])
            cnt = cnt.index_add(0, edges[:, a], one)
        return n / cnt.clamp(min=1)

    def normal_energy(p):
        v = p[sphere_tri.long()]
        n = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
        n = torch.nn.functional.normalize(n, dim=1, eps=1e-12)
        return (1.0 - (n[pairs[:, 0]] * n[pairs[:, 1]]).sum(dim=1)).sum()

    with torch.no_grad():
        save_image(out_dir / "initial.png", tile_images(
            [hard_mask(pos, sphere_tri, m) for m in mvps], 4))

    t0 = time.perf_counter()
    for it in range(args.iters):
        opt.zero_grad(set_to_none=True)
        loss = data_loss()
        loss.backward()
        # C++-style regularization: add the regularizer gradients scaled
        # relative to the data-gradient RMS (weights transfer from C++)
        g_nrm = torch.autograd.grad(normal_energy(pos), pos)[0]
        with torch.no_grad():
            g = pos.grad
            d_rms = rms(g)
            g_lap = pos - nbr_mean(pos)
            g += g_lap * (d_rms / rms(g_lap) * args.laplacian)
            g += g_nrm * (d_rms / rms(g_nrm) * args.normal)
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {loss.item():.6f}")
    dt = (time.perf_counter() - t0) / args.iters * 1e3
    print(f"{dt:.2f} ms/iter over {args.iters} iters")

    # Final silhouette IoU against hard target masks (all views)
    with torch.no_grad():
        hard_targets = [hard_mask(bunny_pos, bunny_tri, m) for m in mvps]
        finals = [hard_mask(pos, sphere_tri, m) for m in mvps]
        inter = sum(((f > 0.5) & (t > 0.5)).sum() for f, t in
                    zip(finals, hard_targets))
        union = sum(((f > 0.5) | (t > 0.5)).sum() for f, t in
                    zip(finals, hard_targets))
        print(f"silhouette IoU: {inter.item() / union.item():.4f}")
        save_image(out_dir / "final.png", tile_images(finals, 4))
        save_obj(out_dir / "final_mesh.obj", pos, sphere_tri)
    print(f"outputs in {out_dir}/ (targets/initial/final + final_mesh.obj)")


if __name__ == "__main__":
    main()
