"""Multi-view texture optimization (Python port of
examples/texture_optimization): recover the bunny's texture atlas from
unlit renderings, starting from a gray texture.

Per view: rasterize -> interpolate(uv) -> texture(tex); the target is the
same pipeline sampling the ground-truth atlas. Optional sub-pixel camera
jitter (the C++ example's Sobol countermeasure, Halton here) sweeps pixel
centers across the surface so fresh texels are constrained every
iteration. Adam runs on the texture. Runs on either backend:

    python examples/python/texture_optimization.py                      # dressi (Vulkan)
    python examples/python/texture_optimization.py --backend nvdiffrast # CUDA
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from common import (
    add_common_args,
    circle_mvps,
    default_device,
    halton,
    load_obj,
    resolve_backend,
    save_image,
    tile_images,
    transform_clip,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--data", default="data/bunny")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--tex-res", type=int, default=512,
                    help="optimized atlas resolution")
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--no-jitter", action="store_true",
                    help="disable the sub-pixel camera jitter")
    ap.add_argument("--out", default="texopt_py_out")
    return ap.parse_args()


def main():
    args = parse_args()
    args.iters = args.iters or 500
    res = args.res or 256
    device = torch.device(args.device) if args.device else default_device()
    dr, ctx = resolve_backend(args.backend)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos, uv, tri = load_obj(pathlib.Path(args.data) / "bunny.obj")
    pos, uv, tri = pos.to(device), uv.to(device), tri.to(device)

    from PIL import Image

    atlas = Image.open(pathlib.Path(args.data) / "bunny-atlas.jpg")
    atlas = atlas.resize((args.tex_res, args.tex_res))
    gt_tex = torch.tensor(np.asarray(atlas, np.float32) / 255.0,
                          device=device)
    print(f"mesh: {len(pos)}v/{len(tri)}f  atlas: {args.tex_res}^2  "
          f"backend={args.backend} device={device.type} "
          f"{res}x{res} x{args.views} views")

    mvps = circle_mvps(args.views, device=device)

    def render(tex, mvp, jitter):
        clip = transform_clip(pos, mvp)
        if jitter is not None:
            # Sub-pixel projection offset (TAA style): pixel centers sweep
            # their surface footprint while colors stay glued to geometry
            j = torch.zeros_like(clip)
            j[:, 0] = jitter[0] * clip[:, 3]
            j[:, 1] = jitter[1] * clip[:, 3]
            clip = clip + j
        rast, _ = dr.rasterize(ctx, clip[None], tri, resolution=[res, res])
        mask = (rast[..., 3:] > 0).float()
        uv_img, _ = dr.interpolate(uv, rast, tri)
        # nvdiffrast requires a batched texture [N,H,W,C]
        return mask * dr.texture(tex[None], uv_img, filter_mode="nearest")

    tex = torch.full((args.tex_res, args.tex_res, 3), 0.5, device=device,
                     requires_grad=True)
    opt = torch.optim.Adam([tex], lr=args.lr)

    with torch.no_grad():
        save_image(out_dir / "targets.png", tile_images(
            [render(gt_tex, m, None)[0] for m in mvps], 3))

    t0 = time.perf_counter()
    for it in range(args.iters):
        jitter = None
        if not args.no_jitter:
            jitter = ((halton(it + 1, 2) - 0.5) * 2.0 / res,
                      (halton(it + 1, 3) - 0.5) * 2.0 / res)
        opt.zero_grad(set_to_none=True)
        loss = tex.new_zeros(())
        for m in mvps:
            with torch.no_grad():
                target = render(gt_tex, m, jitter)
            pred = render(tex, m, jitter)
            loss = loss + ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  loss {loss.item():.6f}")
    dt = (time.perf_counter() - t0) / args.iters * 1e3
    print(f"{dt:.2f} ms/iter over {args.iters} iters")

    with torch.no_grad():
        finals, psnrs = [], []
        for m in mvps:
            target = render(gt_tex, m, None)
            pred = render(tex, m, None)
            finals.append(pred[0])
            mse = ((pred - target) ** 2).mean().item()
            psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        print(f"rendered-view PSNR: {np.mean(psnrs):.2f} dB "
              f"(per view: {', '.join(f'{p:.1f}' for p in psnrs)})")
        save_image(out_dir / "final_views.png", tile_images(finals, 3))
        # Atlas images are texture-space (row 0 = top already): no view flip
        save_image(out_dir / "recovered_atlas.png", tex.clamp(0, 1),
                   flip_v=False)
        save_image(out_dir / "gt_atlas.png", gt_tex, flip_v=False)
    print(f"outputs in {out_dir}/")


if __name__ == "__main__":
    main()
