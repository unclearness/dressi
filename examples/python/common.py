"""Shared utilities for the Python examples.

Only `dressi.torch` / `nvdiffrast.torch` and plain torch/numpy are used —
mesh loading, cameras, regularizers and image IO all live here as torch /
Python code so the examples read like ordinary nvdiffrast programs.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_backend(name: str):
    """Returns (dr_module, ctx). The dressi API is nvdiffrast drop-in, so
    the examples run unchanged on either backend."""
    if name == "nvdiffrast":
        # Lenient MSVC banner decoding for the JIT plugin build (the 'oem'
        # codec is broken on Japanese-locale Windows)
        import torch.utils.cpp_extension as _ce

        _ce.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
        # Recent torch no longer puts the JIT build dir on sys.path, which
        # nvdiffrast 0.3.3's importlib.import_module relies on
        for _name in ("nvdiffrast_plugin", "nvdiffrast_plugin_gl"):
            _d = _ce._get_build_directory(_name, False)
            if _d not in sys.path:
                sys.path.insert(0, _d)
        import nvdiffrast.torch as dr

        return dr, dr.RasterizeCudaContext()
    if name == "dressi":
        sys.path.insert(0, str(_REPO_ROOT / "python"))
        import dressi.torch as dr

        return dr, dr.RasterizeVulkanContext()
    raise ValueError(f"unknown backend '{name}' (dressi|nvdiffrast)")


def add_common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--backend", default="dressi",
                    choices=["dressi", "nvdiffrast"])
    ap.add_argument("--device", default=None,
                    help="torch device (default: cuda if available)")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--res", type=int, default=None)


# ------------------------------- assets ------------------------------------


def load_obj(path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimal OBJ loader matching the C++ examples' LoadObjMesh: unwelds
    one vertex per unique (v, vt) pair, flips vt.y (OBJ origin is
    bottom-left, image row 0 is the top), normalizes to center origin /
    max extent 1. Returns (pos [V,3] f32, uv [V,2] f32, tri [F,3] i32)."""
    vs, vts = [], []
    unique: dict[tuple[int, int], int] = {}
    pos, uv, idx = [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            vs.append([float(t[1]), float(t[2]), float(t[3])])
        elif t[0] == "vt":
            vts.append([float(t[1]), float(t[2])])
        elif t[0] == "f":
            corners = []
            for w in t[1:]:
                parts = w.split("/")
                vi = int(parts[0]) - 1
                ti = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                key = (vi, ti)
                if key not in unique:
                    unique[key] = len(pos)
                    pos.append(vs[vi])
                    uv.append([vts[ti][0], 1.0 - vts[ti][1]]
                              if ti >= 0 else [0.0, 0.0])
                corners.append(unique[key])
            for k in range(1, len(corners) - 1):  # fan-triangulate
                idx.append([corners[0], corners[k], corners[k + 1]])
    p = np.asarray(pos, np.float32)
    mins, maxs = p.min(0), p.max(0)
    p = (p - 0.5 * (mins + maxs)) / (maxs - mins).max()
    return (torch.from_numpy(p), torch.tensor(uv, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.int32))


def generate_icosphere(level: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Welded unit icosphere (port of the C++ example generator)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.asarray(v, np.float64) / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        midpoint: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                midpoint[key] = len(verts) - 1
            return midpoint[key]

        nxt = []
        for f in faces:
            a, b, c = mid(f[0], f[1]), mid(f[1], f[2]), mid(f[2], f[0])
            nxt += [(f[0], a, c), (f[1], b, a), (f[2], c, b), (a, b, c)]
        faces = nxt
    return (torch.tensor(np.asarray(verts), dtype=torch.float32),
            torch.tensor(faces, dtype=torch.int32))


# ------------------------------- cameras -----------------------------------


def look_at(eye, center, up) -> np.ndarray:
    eye, center, up = (np.asarray(v, np.float32) for v in (eye, center, up))
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective_gl(fovy_rad: float, aspect: float, near: float,
                   far: float) -> np.ndarray:
    t = 1.0 / np.tan(fovy_rad / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def circle_mvps(n_views: int, radius=1.6, height=0.5, fovy_deg=45.0,
                device=None) -> list[torch.Tensor]:
    """The C++ examples' camera ring: n views on a circle, slightly
    elevated, looking at the origin."""
    proj = perspective_gl(np.deg2rad(fovy_deg), 1.0, 0.1, 10.0)
    mvps = []
    for v in range(n_views):
        ang = 2.0 * np.pi * v / n_views
        eye = (radius * np.sin(ang), height, radius * np.cos(ang))
        mvps.append(torch.tensor(proj @ look_at(eye, (0, 0, 0), (0, 1, 0)),
                                 device=device))
    return mvps


def transform_clip(pos: torch.Tensor, mvp: torch.Tensor) -> torch.Tensor:
    """[V,3] object-space -> [V,4] clip-space (differentiable)."""
    posh = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1)
    return (mvp.to(pos) @ posh.t()).t().contiguous()


def transform_clip_batched(pos: torch.Tensor,
                           mvps: torch.Tensor) -> torch.Tensor:
    """Project [V,3] positions through N cameras at once -> [N,V,4].

    One batched matmul instead of a Python per-view loop (the loop's
    torch.cat / ones_like / stack dominate the host time in a small-mesh
    silhouette step); pass a stacked [N,4,4] mvp tensor.
    """
    posh = torch.cat([pos, torch.ones_like(pos[:, :1])], dim=1)  # [V,4]
    # [N,4,4] @ [4,V] -> [N,4,V] -> [N,V,4]
    return (mvps.to(pos) @ posh.t()).transpose(1, 2).contiguous()


# ----------------------------- regularizers --------------------------------


def build_edges(tri: torch.Tensor) -> torch.Tensor:
    """Unique undirected edges [E,2] (int64) from the face list."""
    e = torch.cat([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 0).long()
    e = torch.sort(e, dim=1).values
    return torch.unique(e, dim=0)


def laplacian_energy(pos: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Uniform-Laplacian energy: mean ||pos - mean(neighbors)||^2."""
    n = pos.shape[0]
    nbr_sum = torch.zeros_like(pos)
    cnt = torch.zeros(n, 1, dtype=pos.dtype, device=pos.device)
    for a, b in ((0, 1), (1, 0)):
        nbr_sum = nbr_sum.index_add(0, edges[:, a], pos[edges[:, b]])
        cnt = cnt.index_add(0, edges[:, a],
                            torch.ones(edges.shape[0], 1, dtype=pos.dtype,
                                       device=pos.device))
    lap = pos - nbr_sum / cnt.clamp(min=1)
    return (lap ** 2).sum(dim=1).mean()


def build_face_pairs(tri: torch.Tensor) -> torch.Tensor:
    """Edge-adjacent face pairs [P,2] (int64)."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    tri_np = tri.cpu().numpy()
    for fi, f in enumerate(tri_np):
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
        pass
    pairs = []
    for fs in edge_faces.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                pairs.append((fs[i], fs[j]))
    return torch.tensor(pairs, dtype=torch.int64, device=tri.device)


def normal_consistency_energy(pos: torch.Tensor, tri: torch.Tensor,
                              pairs: torch.Tensor) -> torch.Tensor:
    """mean(1 - n_f . n_g) over edge-adjacent face pairs."""
    v = pos[tri.long()]
    n = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
    n = torch.nn.functional.normalize(n, dim=1, eps=1e-12)
    return (1.0 - (n[pairs[:, 0]] * n[pairs[:, 1]]).sum(dim=1)).mean()


def save_obj(path, pos: torch.Tensor, tri: torch.Tensor):
    """Writes positions/faces as a Wavefront OBJ (uv omitted)."""
    with open(path, "w", encoding="utf-8") as f:
        for v in pos.detach().cpu().tolist():
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in tri.cpu().tolist():
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


# ------------------------------- image IO ----------------------------------


def halton(i: int, base: int) -> float:
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def save_image(path, img: torch.Tensor, flip_v: bool = True):
    """img: [H,W] or [H,W,C] float in [0,1]. Row 0 is NDC y=-1 (OpenGL
    bottom), so images are flipped vertically for viewing by default —
    the same convention nvdiffrast's samples use."""
    from PIL import Image

    a = img.detach().cpu().clamp(0, 1).numpy()
    if flip_v:
        a = a[::-1]
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(str(path))


def tile_images(imgs: list[torch.Tensor], cols: int) -> torch.Tensor:
    """Same-size [H,W,C] images tiled row-major into one grid."""
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[0], imgs[0].shape[1]
    c = imgs[0].shape[2] if imgs[0].dim() == 3 else 1
    grid = torch.zeros(rows * h, cols * w, c)
    for i, im in enumerate(imgs):
        if im.dim() == 2:
            im = im[..., None]
        r, q = divmod(i, cols)
        grid[r * h:(r + 1) * h, q * w:(q + 1) * w] = im.detach().cpu()
    return grid
