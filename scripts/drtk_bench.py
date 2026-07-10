# DRTK (github.com/facebookresearch/DRTK) silhouette-optimization benchmark
# on the glTF Avocado (406v/682f). Mirrors bench.py (nvdiffrast) exactly:
# per iteration rasterize + boundary-gradient mask + MSE + backward to
# vertex positions + Adam step. Warmup 50, measure 200.
import json
import os
import sys
import time

import numpy as np
import torch
import drtk

GLTF = r"E:\Dev\dressi2\data\glTF-Sample-Models-main\glTF-Sample-Models-main\2.0\Avocado\glTF\Avocado.gltf"

COMP_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
COMP_NP = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_N = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_accessor(gltf, buf, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    n = TYPE_N[acc["type"]]
    dt = COMP_NP[acc["componentType"]]
    itemsize = COMP_SIZE[acc["componentType"]] * n
    stride = bv.get("byteStride", itemsize)
    ofs = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = np.zeros((acc["count"], n), dtype=dt)
    for i in range(acc["count"]):
        out[i] = np.frombuffer(buf, dtype=dt, count=n, offset=ofs + i * stride)
    return out


def load_avocado():
    gltf = json.load(open(GLTF))
    buf = open(os.path.join(os.path.dirname(GLTF),
                            gltf["buffers"][0]["uri"]), "rb").read()
    prim = gltf["meshes"][0]["primitives"][0]
    pos = read_accessor(gltf, buf, prim["attributes"]["POSITION"]).astype(np.float32)
    tri = read_accessor(gltf, buf, prim["indices"]).reshape(-1, 3).astype(np.int32)
    mins, maxs = pos.min(0), pos.max(0)
    pos = (pos - 0.5 * (mins + maxs)) / max(maxs - mins)
    return pos, tri


def look_at(eye, center, up):
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fovy, aspect, near, far):
    t = 1.0 / np.tan(fovy / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def main():
    dev = torch.device("cuda")
    pos_np, tri_np = load_avocado()
    print(f"avocado: {len(pos_np)} verts, {len(tri_np)} faces  backend=drtk")
    mvp = perspective(np.deg2rad(45), 1.0, 0.1, 10.0) @ look_at(
            np.array([0.0, 0.55, 1.6], np.float32),
            np.zeros(3, np.float32), np.array([0.0, 1.0, 0.0], np.float32))
    mvp_t = torch.tensor(mvp, device=dev)
    tri = torch.tensor(tri_np, device=dev)  # int32 [F,3]
    pos_ref = torch.tensor(pos_np, device=dev)

    def render_mask(pos, res, with_grad):
        posh = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1)
        clip = (mvp_t @ posh.t()).t()
        # clip -> DRTK pixel space: x,y in pixels (y down), z = view depth
        w = clip[:, 3:4]
        sx = (clip[:, 0:1] / w * 0.5 + 0.5) * res
        sy = (1.0 - (clip[:, 1:2] / w * 0.5 + 0.5)) * res
        v_pix = torch.cat([sx, sy, w], dim=1)[None]  # [1,V,3]
        index_img = drtk.rasterize(v_pix, tri, height=res, width=res)
        mask = (index_img != -1).unsqueeze(1).float()  # [1,1,H,W]
        if not with_grad:
            return mask
        _, bary_img = drtk.render(v_pix, tri, index_img)
        img = drtk.edge_grad_estimator(
            v_pix=v_pix, vi=tri, bary_img=bary_img, img=mask,
            index_img=index_img)
        return img

    for res in (256, 512, 1024, 2048, 4096):
        with torch.no_grad():
            target = render_mask(pos_ref, res, False).detach()
        pos = (pos_ref * 0.8).clone().requires_grad_(True)
        opt = torch.optim.Adam([pos], lr=0.01)
        n_warm, n_meas = 50, 200

        def step():
            opt.zero_grad(set_to_none=True)
            img = render_mask(pos, res, True)
            loss = ((img - target) ** 2).mean()
            loss.backward()
            opt.step()
            return loss

        for _ in range(n_warm):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_meas):
            loss = step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n_meas * 1e3
        print(f"drtk {res}x{res}: {ms:.3f} ms/iter"
              f"  (final loss {loss.item():.6f})")


if __name__ == "__main__":
    main()
