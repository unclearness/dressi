# nvdiffrast silhouette-optimization benchmark on the glTF Avocado
# (406v/682f, the Dressi paper's Table 4 mesh). Mirrors the Dressi
# silhouette example: per iteration rasterize + antialias(mask) + MSE +
# backward to vertex positions + Adam step.
import json
import struct
import sys
import time

import numpy as np
import torch
import torch.utils.cpp_extension as _ce

# The 'oem' codec is broken on this Japanese-locale Windows; decode MSVC
# banners leniently so the JIT plugin build can proceed
_ce.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

# torch 2.13's load() no longer puts the JIT build dir on sys.path, which
# nvdiffrast 0.3.3's importlib.import_module relies on
import os
_cache = os.path.join(os.environ.get("LOCALAPPDATA", ""),
                      "torch_extensions", "torch_extensions", "Cache",
                      "py312_cu130")
for _name in ("nvdiffrast_plugin_gl", "nvdiffrast_plugin"):
    _ext = os.path.join(_cache, _name)
    if os.path.isdir(_ext):
        sys.path.insert(0, _ext)

import nvdiffrast.torch as dr  # noqa: E402

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
    import os
    buf = open(os.path.join(os.path.dirname(GLTF),
                            gltf["buffers"][0]["uri"]), "rb").read()
    prim = gltf["meshes"][0]["primitives"][0]
    pos = read_accessor(gltf, buf, prim["attributes"]["POSITION"]).astype(np.float32)
    tri = read_accessor(gltf, buf, prim["indices"]).reshape(-1, 3).astype(np.int32)
    # normalize: center origin, max extent 1 (same as the Dressi example)
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
    backend = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    dev = torch.device("cuda")
    pos_np, tri_np = load_avocado()
    print(f"avocado: {len(pos_np)} verts, {len(tri_np)} faces"
          f"  backend={backend}")
    mvp = perspective(np.deg2rad(45), 1.0, 0.1, 10.0) @ look_at(
            np.array([0.0, 0.55, 1.6], np.float32),
            np.zeros(3, np.float32), np.array([0.0, 1.0, 0.0], np.float32))
    mvp_t = torch.tensor(mvp, device=dev)
    tri = torch.tensor(tri_np, device=dev)
    pos_ref = torch.tensor(pos_np, device=dev)
    ctx = (dr.RasterizeGLContext()
           if backend == "gl" else dr.RasterizeCudaContext())

    def render_mask(pos, res):
        posh = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1)
        clip = (mvp_t @ posh.t()).t().contiguous()[None]
        rast, _ = dr.rasterize(ctx, clip, tri, resolution=[res, res])
        mask = (rast[..., 3:] > 0).float()
        return dr.antialias(mask, rast, clip, tri), mask

    for res in (256, 512, 1024, 2048, 4096):
        with torch.no_grad():
            target = render_mask(pos_ref, res)[1].detach()  # hard GT mask
        pos = (pos_ref * 0.8).clone().requires_grad_(True)
        opt = torch.optim.Adam([pos], lr=0.01)
        n_warm, n_meas = 50, 200

        def step():
            opt.zero_grad(set_to_none=True)
            aa, _ = render_mask(pos, res)
            loss = ((aa - target) ** 2).mean()
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
        print(f"nvdiffrast[{backend}] {res}x{res}: {ms:.3f} ms/iter"
              f"  (final loss {loss.item():.6f})")


if __name__ == "__main__":
    main()
