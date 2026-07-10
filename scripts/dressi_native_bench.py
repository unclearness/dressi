# Fused-pipeline benchmark: the SAME silhouette workload as
# dressi_torch_bench.py / nvdiffrast_bench.py, but built as ONE Dressi
# graph via the native API (dressi._C) -- rasterize + antialias + MSE +
# in-graph Adam all execute inside a single cached Vulkan command buffer,
# with zero per-iteration PCIe traffic. This is the "escape hatch" that
# recovers C++-level speed from Python when the eager nvdiffrast-style
# API's per-op transfers dominate. Run with PYTHONPATH=python.
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))
import dressi.torch as dr  # noqa: E402
from dressi import _C  # noqa: E402
from dressi_torch_bench import load_avocado, look_at, perspective  # noqa: E402

f = _C.f


def main():
    resolutions = [int(a) for a in sys.argv[1:]] or [256, 512, 1024]
    pos_np, tri_np = load_avocado()
    n_verts, n_faces = len(pos_np), len(tri_np)
    print(f"avocado: {n_verts} verts, {n_faces} faces  backend=dressi(native fused)")
    mvp = perspective(np.deg2rad(45), 1.0, 0.1, 10.0) @ look_at(
            np.array([0.0, 0.55, 1.6], np.float32),
            np.zeros(3, np.float32), np.array([0.0, 1.0, 0.0], np.float32))
    tri = torch.tensor(tri_np)
    pos_ref = torch.tensor(pos_np)
    tctx = dr.RasterizeVulkanContext()

    for res in resolutions:
        # Target mask rendered once through the torch layer
        with torch.no_grad():
            posh = torch.cat([pos_ref, torch.ones_like(pos_ref[:, :1])], 1)
            clip_t = (torch.tensor(mvp) @ posh.t()).t().contiguous()
            rast, _ = dr.rasterize(tctx, clip_t, tri, resolution=[res, res])
            target_img = (rast[..., 3:] > 0).float()[0].numpy()

        # ----- one fused graph: transform + raster + AA + MSE + Adam ------
        md = tctx.mesh(tri)
        pos = _C.Variable(_C.VType.VEC3, n_verts, 1)
        ones = _C.Variable(_C.VType.FLOAT, n_verts, 1)
        faces_i = _C.Variable(_C.VType.IVEC3, n_faces, 1)
        faces_f = _C.Variable(_C.VType.VEC3, n_faces, 1)
        vft = _C.Variable(_C.VType.FLOAT, n_verts, md.max_deg)
        target = _C.Variable(_C.VType.FLOAT, res, res)
        adam_m = _C.Variable(_C.VType.VEC3, n_verts, 1)
        adam_v = _C.Variable(_C.VType.VEC3, n_verts, 1)
        adam_t = _C.Variable(_C.VType.FLOAT, 1, 1)

        # clip = mvp @ (pos, 1) with column constants (in-graph, fusable)
        col = [f.vec4(*mvp[:, c].tolist()) for c in range(4)]
        clip = (col[0] * f.get_x(pos) + col[1] * f.get_y(pos)
                + col[2] * f.get_z(pos) + col[3])
        mask = f.rasterize(clip, ones, faces_i, res, res)
        tid = f.rasterize_face_id(clip, ones, faces_i, res, res)
        pred = f.anti_alias(mask, tid, clip, faces_f, vft, adam_t, 8)
        diff = f.stop_gradient(target) - pred
        loss = diff * diff * (1.0 / float(res * res))
        pos.set_requires_grad_recursively(True)

        ad = _C.DressiAD(tctx._ctx)
        ad.set_loss_var(loss)
        state = {"registered": False}

        def optimizer(xs, gxs):
            b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.01
            g = gxs[0]
            t_new = adam_t + 1.0
            m_new = adam_m * b1 + g * (1.0 - b1)
            v_new = adam_v * b2 + g * g * (1.0 - b2)
            bc1 = 1.0 - f.pow(f.constant(b1), t_new)
            bc2 = 1.0 - f.pow(f.constant(b2), t_new)
            step = (m_new / bc1) / (f.sqrt(v_new / bc2) + eps) * lr
            if not state["registered"]:
                ad.add_update(adam_m, m_new)
                ad.add_update(adam_v, v_new)
                ad.add_update(adam_t, t_new)
                state["registered"] = True
            return [xs[0] - step]

        ad.set_optimizer(optimizer)
        ad.mark_output(loss)

        ad.send_img(pos, (pos_np * 0.8).reshape(1, n_verts, 3))
        ad.send_img(ones, np.ones((1, n_verts, 1), np.float32))
        ad.send_img(faces_i, md.faces_f32)
        ad.send_img(faces_f, md.faces_f32)
        ad.send_img(vft, md.vtx_faces_tex)
        ad.send_img(target, target_img)
        ad.send_img(adam_m, np.zeros((1, n_verts, 3), np.float32))
        ad.send_img(adam_v, np.zeros((1, n_verts, 3), np.float32))
        ad.send_img(adam_t, np.zeros((1, 1, 1), np.float32))

        n_warm, n_meas = 50, 200
        for _ in range(n_warm):
            ad.exec_step()
        t0 = time.perf_counter()
        for _ in range(n_meas):
            ad.exec_step()
        ms = (time.perf_counter() - t0) / n_meas * 1e3
        final_loss = ad.recv_img(loss).sum()
        print(f"dressi.native {res}x{res}: {ms:.3f} ms/iter"
              f"  (final loss {final_loss:.6f})")


if __name__ == "__main__":
    main()
