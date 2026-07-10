"""nvdiffrast-compatible ops backed by shape-keyed Dressi engines."""

from __future__ import annotations

import numpy as np
import torch

from .. import _C
from .context import RasterizeVulkanContext
from .convert import (
    as_hwc,
    check_resolution,
    clip_gl_to_vk,
    clip_grad_vk_to_gl,
    split_batch,
    to_cpu_f32,
    vertex_array,
)
from .engine import Engine, upload_token
from .provenance import InterpInfo, RastInfo, attach, lookup

f = _C.f

# Per-corner barycentric constants (u, v) = (1,0), (0,1), (0,0): rasterizing
# them as unwelded vertex attributes yields nvdiffrast's perspective-correct
# barycentrics per pixel.
_CORNER_UV = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

_VTYPE_BY_CHANNELS = {
    1: _C.VType.FLOAT,
    2: _C.VType.VEC2,
    3: _C.VType.VEC3,
    4: _C.VType.VEC4,
}

_ZERO_SEED = np.zeros((1, 1, 1), np.float32)


def _vtype_of(channels: int, what: str):
    vt = _VTYPE_BY_CHANNELS.get(channels)
    if vt is None:
        raise ValueError(
            f"dressi.torch: {what} must have 1..4 channels, got {channels}")
    return vt


def _needs_grad(*tensors) -> bool:
    return torch.is_grad_enabled() and any(
        isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)


def _attr_item(attr_b: torch.Tensor, n: int) -> torch.Tensor:
    """Batch-aware view: [V, C] is shared across items, [N, V, C] indexes."""
    return attr_b if attr_b.dim() == 2 else attr_b[n]


def _stack_batch(outs: list) -> torch.Tensor:
    """torch.stack copies; a single item just gets a batch-dim view."""
    return outs[0].unsqueeze(0) if len(outs) == 1 else torch.stack(outs, 0)


def _image_batch(t: torch.Tensor, name: str) -> torch.Tensor:
    """Accept [H, W, C] or [N, H, W, C]; return [N, H, W, C]."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 4:
        return t
    raise ValueError(
        f"{name} must be [H, W, C] or [N, H, W, C], got {tuple(t.shape)}")


# ------------------------------- rasterize ---------------------------------


def _build_rasterize_engine(ctx: RasterizeVulkanContext, n_faces: int,
                            h: int, w: int) -> Engine:
    eng = Engine(ctx._ctx)
    clip3 = eng.leaf("clip3", _C.VType.VEC4, n_faces * 3, 1)
    attr4 = eng.leaf("attr4", _C.VType.VEC4, n_faces * 3, 1)
    faces_seq = eng.leaf("faces_seq", _C.VType.IVEC3, n_faces, 1)
    ones3f = eng.leaf("ones3f", _C.VType.FLOAT, n_faces * 3, 1)

    g = f.rasterize(clip3, attr4, faces_seq, w, h)  # (u, v, z_gl, w) persp-correct
    tid = f.rasterize_face_id(clip3, ones3f, faces_seq, w, h)
    mask = f.greater(tid, f.constant(0.5))
    zw = f.get_z(g) / (f.get_w(g) + (1.0 - mask))  # guarded z_gl/w (bg -> 0)
    rast = f.vec4v(f.get_x(g), f.get_y(g), zw, tid)
    eng.finalize(rast)
    return eng


def rasterize(ctx: RasterizeVulkanContext, pos: torch.Tensor,
              tri: torch.Tensor, resolution, ranges=None, grad_db: bool = False):
    """Drop-in nvdiffrast rasterize: returns (rast, rast_db).

    rast[..., :] = (u, v, z/w, triangle_id + 1); 0 = background. v1 is not
    differentiable w.r.t. pos (geometry gradients come from antialias /
    rasterize_soft, matching the silhouette workflow); rast_db is zeros.
    """
    if ranges is not None:
        raise NotImplementedError("dressi.torch: range mode is not supported")
    del grad_db
    h, w = check_resolution(resolution)
    in_device = pos.device
    pos_b = split_batch(to_cpu_f32(pos.detach(), "pos"))
    md = ctx.mesh(tri)
    if pos_b.shape[2] != 4:
        raise ValueError(f"pos must have 4 components, got {pos_b.shape[2]}")

    key = ("rasterize", h, w, md.n_faces)
    eng: Engine = ctx.engine(
        key, lambda: _build_rasterize_engine(ctx, md.n_faces, h, w))

    with ctx.lock:
        eng.upload("faces_seq", md.faces_seq_f32, token=("static", id(md)),
                   ref=md)
        eng.upload("ones3f", np.ones((1, md.n_faces * 3, 1), np.float32),
                   token=("static", id(md)), ref=md)
        outs = []
        for n in range(pos_b.shape[0]):
            pos_n = pos_b[n]
            clip3 = clip_gl_to_vk(pos_n)[md.unweld]  # [3F, 4] Vulkan depth
            uv = torch.from_numpy(np.tile(_CORNER_UV, (md.n_faces, 1)))
            attr4 = torch.cat([uv, pos_n[md.unweld][:, 2:4]], dim=1)
            tok = (upload_token(pos_b), id(md), n)
            eng.upload("clip3", vertex_array(clip3), token=("clip3",) + tok,
                       ref=pos_b)
            eng.upload("attr4", vertex_array(attr4), token=("attr4",) + tok,
                       ref=pos_b)
            eng.run()
            outs.append(torch.from_numpy(eng.read_output()))

    rast = _stack_batch(outs).to(in_device)
    attach(rast, RastInfo(pos=pos_b, tri=tri, resolution=(h, w),
                          extra={"ctx": ctx}))
    return rast, torch.zeros_like(rast)


# ------------------------------ interpolate --------------------------------


def _build_interpolate_engine(ctx, md, h: int, w: int, c_attr: int,
                              train: bool) -> Engine:
    eng = Engine(ctx._ctx)
    attr_l = eng.leaf("attr", _vtype_of(c_attr, "attr"), md.n_verts, 1)
    rast_l = eng.leaf("rast", _C.VType.VEC4, w, h)
    clip_l = eng.leaf("clip", _C.VType.VEC4, md.n_verts, 1)
    faces_l = eng.leaf("faces_f", _C.VType.VEC3, md.n_faces, 1)
    vft_l = eng.leaf("vft", _C.VType.FLOAT, md.n_verts, md.max_deg)
    seed_l = eng.leaf("ffseed", _C.VType.FLOAT, 1, 1)

    u, v = f.get_x(rast_l), f.get_y(rast_l)
    tid = f.get_w(rast_l)
    prj = [f.lookup_faces(clip_l, faces_l, vft_l, c) for c in range(3)]
    # Exact FaceFetch backward (n_samples=0): the true VJP for interior
    # support, chained with LookupFaces' exact vertex scatter
    pix = [
        f.face_fetch(f.lookup_faces(attr_l, faces_l, vft_l, c), tid, prj[0],
                     prj[1], prj[2], seed_l, 1.0, 0)
        for c in range(3)
    ]
    out = u * pix[0] + v * pix[1] + (1.0 - u - v) * pix[2]
    eng.finalize(out, grad_leaves=["attr"] if train else [])
    return eng


def _interpolate_uploads(eng: Engine, md, rinfo: RastInfo, rast_cpu, attr_b,
                         n: int):
    eng.upload("faces_f", md.faces_f32, token=("static", id(md)), ref=md)
    eng.upload("vft", md.vtx_faces_tex, token=("vft", id(md)), ref=md)
    eng.upload("ffseed", _ZERO_SEED, token="zero")
    pos_n = rinfo.pos[n if rinfo.pos.shape[0] > 1 else 0]
    eng.upload("clip", vertex_array(clip_gl_to_vk(pos_n)),
               token=("clip", upload_token(rinfo.pos), n), ref=rinfo.pos)
    eng.upload("rast", as_hwc(rast_cpu[n]),
               token=("rast", upload_token(rast_cpu), n), ref=rast_cpu)
    attr_n = _attr_item(attr_b, n)
    eng.upload("attr", vertex_array(attr_n),
               token=("attr", upload_token(attr_b), n), ref=attr_b)


class _InterpolateFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, attr, rast, vctx, md, rinfo, train):
        h, w = rinfo.resolution
        attr_b = to_cpu_f32(attr.detach(), "attr")
        rast_cpu = to_cpu_f32(rast.detach(), "rast")
        c_attr = attr_b.shape[-1]
        n_batch = rast_cpu.shape[0]

        key = ("interpolate", h, w, md.n_faces, md.n_verts, c_attr, train)
        eng: Engine = vctx.engine(
            key,
            lambda: _build_interpolate_engine(vctx, md, h, w, c_attr, train))

        outs = []
        with vctx.lock:
            for n in range(n_batch):
                _interpolate_uploads(eng, md, rinfo, rast_cpu, attr_b, n)
                eng.run()
                outs.append(torch.from_numpy(eng.read_output()))

        fctx.dressi = (vctx, eng, md, rinfo, n_batch, attr.dim())
        fctx.save_for_backward(attr, rast)
        return _stack_batch(outs)

    @staticmethod
    def backward(fctx, grad_out):
        vctx, eng, md, rinfo, n_batch, attr_dim = fctx.dressi
        attr, rast = fctx.saved_tensors
        attr_b = to_cpu_f32(attr.detach(), "attr")
        rast_cpu = to_cpu_f32(rast.detach(), "rast")
        g = to_cpu_f32(grad_out, "grad_out")

        grads = []
        with vctx.lock:
            for n in range(n_batch):
                _interpolate_uploads(eng, md, rinfo, rast_cpu, attr_b, n)
                eng.upload("__seed__", as_hwc(g[n]))
                eng.run()
                grads.append(torch.from_numpy(eng.read_grad("attr"))[0])
        g_attr = (torch.stack(grads, dim=0) if attr_dim == 3
                  else torch.stack(grads, dim=0).sum(dim=0))
        return g_attr.to(attr.device), None, None, None, None, None


def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Drop-in nvdiffrast interpolate: (pixel_attr, None).

    Differentiable w.r.t. `attr` (exact VJP). Gradients w.r.t. `rast`
    (i.e. barycentric derivatives) are not produced in v1; `diff_attrs`
    (attribute pixel derivatives) is unsupported.
    """
    if rast_db is not None or diff_attrs is not None:
        raise NotImplementedError(
            "dressi.torch.interpolate: rast_db/diff_attrs are not supported")
    rinfo: RastInfo = lookup(rast, "rasterize")
    vctx = rinfo.extra["ctx"]  # the context travels with the provenance
    md = vctx.mesh(tri)
    train = _needs_grad(attr)
    out = _InterpolateFn.apply(attr, rast, vctx, md, rinfo, train)
    out = out.to(attr.device)
    attach(out, InterpInfo(attr=attr, rast_info=rinfo))
    return out, None


# -------------------------------- texture ----------------------------------


def _build_texture_engine(ctx, h, w, th, tw, c_tex, mesh_dims, train) -> Engine:
    eng = Engine(ctx._ctx)
    tex_l = eng.leaf("tex", _vtype_of(c_tex, "tex"), tw, th)
    uv_l = eng.leaf("uv", _C.VType.VEC2, w, h)
    if train:
        n_verts, n_faces = mesh_dims
        uvclip_l = eng.leaf("uv_clip", _C.VType.VEC4, n_verts, 1)
        sattr_l = eng.leaf("screen_attr", _C.VType.VEC4, n_verts, 1)
        faces_l = eng.leaf("faces_i", _C.VType.IVEC3, n_faces, 1)
        # Inverse-UV table: the mesh rasterized in UV space carrying its
        # screen positions (F::Texture's gradient gather geometry)
        inv_uv = f.rasterize(uvclip_l, sattr_l, faces_l, tw, th)
    else:
        # Forward never reads inv_uv (it only feeds the backward gather);
        # an unreferenced dummy keeps the graph purely sampling
        inv_uv = _C.Variable(_C.VType.VEC4, tw, th)
    out = f.texture(tex_l, uv_l, inv_uv)
    eng.finalize(out, grad_leaves=["tex"] if train else [])
    return eng


class _TextureFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, tex, uv, vctx, iinfo, train):
        tex_b = to_cpu_f32(tex.detach(), "tex")
        uv_cpu = to_cpu_f32(uv.detach(), "uv")
        if tex_b.dim() == 4:
            if tex_b.shape[0] != 1:
                raise NotImplementedError(
                    "dressi.torch.texture: per-item texture batches are not "
                    "supported in v1 (pass [H, W, C] or [1, H, W, C])")
            tex_b = tex_b[0]
        th, tw, c_tex = tex_b.shape
        n_batch, h, w = uv_cpu.shape[0], uv_cpu.shape[1], uv_cpu.shape[2]

        md = rinfo = None
        mesh_dims = ()
        if train:
            rinfo = iinfo.rast_info
            md = vctx.mesh(rinfo.tri)
            mesh_dims = (md.n_verts, md.n_faces)
        key = ("texture", h, w, th, tw, c_tex, mesh_dims, train)
        eng: Engine = vctx.engine(
            key, lambda: _build_texture_engine(vctx, h, w, th, tw, c_tex,
                                               mesh_dims, train))

        outs = []
        with vctx.lock:
            for n in range(n_batch):
                _texture_uploads(eng, tex_b, uv_cpu, md, rinfo, iinfo, n,
                                 train, (h, w))
                eng.run()
                outs.append(torch.from_numpy(eng.read_output()))
        fctx.dressi = (vctx, eng, md, rinfo, iinfo, n_batch, train, (h, w))
        fctx.save_for_backward(tex, uv)
        return _stack_batch(outs)

    @staticmethod
    def backward(fctx, grad_out):
        vctx, eng, md, rinfo, iinfo, n_batch, train, res = fctx.dressi
        tex, uv = fctx.saved_tensors
        tex_b = to_cpu_f32(tex.detach(), "tex")
        if tex_b.dim() == 4:
            tex_b = tex_b[0]
        uv_cpu = to_cpu_f32(uv.detach(), "uv")
        g = to_cpu_f32(grad_out, "grad_out")

        g_tex = None
        with vctx.lock:
            for n in range(n_batch):
                _texture_uploads(eng, tex_b, uv_cpu, md, rinfo, iinfo, n,
                                 train, res)
                eng.upload("__seed__", as_hwc(g[n]))
                eng.run()
                item = torch.from_numpy(eng.read_grad("tex"))
                g_tex = item if g_tex is None else g_tex + item
        if tex.dim() == 4:
            g_tex = g_tex.unsqueeze(0)
        return g_tex.to(tex.device), None, None, None, None


def _texture_uploads(eng, tex_b, uv_cpu, md, rinfo, iinfo, n, train, res):
    h, w = res
    eng.upload("tex", as_hwc(tex_b), token=("tex", upload_token(tex_b)),
               ref=tex_b)
    eng.upload("uv", as_hwc(uv_cpu[n]),
               token=("uv", upload_token(uv_cpu), n), ref=uv_cpu)
    if train:
        vtx_uv = to_cpu_f32(iinfo.attr.detach(), "uv attr")
        if vtx_uv.dim() == 3:
            vtx_uv = vtx_uv[n if vtx_uv.shape[0] > 1 else 0]
        eng.upload("uv_clip", _C.uv_as_clip(vertex_array(vtx_uv)),
                   token=("uv_clip", upload_token(iinfo.attr), n),
                   ref=iinfo.attr)
        pos_n = rinfo.pos[n if rinfo.pos.shape[0] > 1 else 0]
        clip_vk = clip_gl_to_vk(pos_n)
        eng.upload("screen_attr",
                   _C.screen_attr(vertex_array(clip_vk), w, h),
                   token=("sattr", upload_token(rinfo.pos), n), ref=rinfo.pos)
        eng.upload("faces_i", md.faces_f32, token=("static", id(md)), ref=md)


def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None,
            filter_mode="auto", boundary_mode="wrap", max_mip_level=None):
    """Drop-in nvdiffrast texture (nearest filtering only in v1).

    Differentiable w.r.t. `tex` (through the inverse-UV table built from
    `uv`'s provenance); not differentiable w.r.t. `uv`. Mipmaps and linear
    filtering are unsupported.
    """
    if uv_da is not None or mip_level_bias is not None or mip is not None:
        raise NotImplementedError("dressi.torch.texture: mipmaps unsupported")
    if filter_mode not in ("auto", "nearest"):
        raise NotImplementedError(
            f"dressi.torch.texture: filter_mode='{filter_mode}' unsupported "
            "(v1 is nearest-only)")
    del boundary_mode, max_mip_level
    train = _needs_grad(tex)
    iinfo = None
    vctx = None
    if train:
        iinfo = lookup(uv, "interpolate")
        vctx = iinfo.rast_info.extra["ctx"]
    else:
        # Forward-only sampling works on any uv; find a context via
        # provenance when available
        try:
            iinfo = lookup(uv, "interpolate")
            vctx = iinfo.rast_info.extra["ctx"]
        except RuntimeError:
            raise RuntimeError(
                "dressi.torch.texture: `uv` must come from "
                "dressi.torch.interpolate in v1 (context discovery)")
    if uv.dim() == 3:
        uv = uv.unsqueeze(0)
    out = _TextureFn.apply(tex, uv, vctx, iinfo, train)
    return out.to(tex.device)


# ------------------------------- antialias ---------------------------------


def _build_antialias_engine(ctx, md, h, w, c_img, grad_mask,
                            n_samples) -> Engine:
    eng = Engine(ctx._ctx)
    img_l = eng.leaf("img", _vtype_of(c_img, "color"), w, h)
    tid_l = eng.leaf("tid", _C.VType.FLOAT, w, h)
    clip_l = eng.leaf("clip", _C.VType.VEC4, md.n_verts, 1)
    faces_l = eng.leaf("faces_f", _C.VType.VEC3, md.n_faces, 1)
    vft_l = eng.leaf("vft", _C.VType.FLOAT, md.n_verts, md.max_deg)
    seed_l = eng.leaf("aaseed", _C.VType.FLOAT, 1, 1)
    out = f.anti_alias(img_l, tid_l, clip_l, faces_l, vft_l, seed_l,
                       n_samples)
    grads = [n for n, wanted in zip(("img", "clip"), grad_mask) if wanted]
    eng.finalize(out, grad_leaves=grads)
    return eng


def _antialias_uploads(eng, md, color_cpu, rast_cpu, pos_b, n, seed_value):
    eng.upload("faces_f", md.faces_f32, token=("static", id(md)), ref=md)
    eng.upload("vft", md.vtx_faces_tex, token=("vft", id(md)), ref=md)
    eng.upload("img", as_hwc(color_cpu[n]),
               token=("img", upload_token(color_cpu), n), ref=color_cpu)
    eng.upload("tid", as_hwc(rast_cpu[n, :, :, 3:4]),
               token=("tid", upload_token(rast_cpu), n), ref=rast_cpu)
    pos_n = pos_b[n if pos_b.shape[0] > 1 else 0]
    eng.upload("clip", vertex_array(clip_gl_to_vk(pos_n)),
               token=("clip", upload_token(pos_b), n), ref=pos_b)
    eng.upload("aaseed", np.full((1, 1, 1), float(seed_value), np.float32))


class _AntiAliasFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, color, rast, pos, vctx, md, grad_mask, n_samples,
                pos_gradient_boost):
        color_cpu = _image_batch(to_cpu_f32(color.detach(), "color"), "color")
        rast_cpu = to_cpu_f32(rast.detach(), "rast")
        pos_b = split_batch(to_cpu_f32(pos.detach(), "pos"))
        n_batch, h, w = color_cpu.shape[0], color_cpu.shape[1], \
            color_cpu.shape[2]
        c_img = color_cpu.shape[3]

        key = ("antialias", h, w, md.n_faces, md.n_verts, c_img, grad_mask,
               n_samples)
        eng: Engine = vctx.engine(
            key, lambda: _build_antialias_engine(vctx, md, h, w, c_img,
                                                 grad_mask, n_samples))
        outs = []
        with vctx.lock:
            for n in range(n_batch):
                _antialias_uploads(eng, md, color_cpu, rast_cpu, pos_b, n,
                                   getattr(eng, "iter_seed", 0))
                eng.run()
                outs.append(torch.from_numpy(eng.read_output()))
        fctx.dressi = (vctx, eng, md, grad_mask, n_batch, pos_gradient_boost)
        fctx.save_for_backward(color, rast, pos)
        return _stack_batch(outs)

    @staticmethod
    def backward(fctx, grad_out):
        vctx, eng, md, grad_mask, n_batch, boost = fctx.dressi
        color, rast, pos = fctx.saved_tensors
        color_cpu = _image_batch(to_cpu_f32(color.detach(), "color"), "color")
        rast_cpu = to_cpu_f32(rast.detach(), "rast")
        pos_b = split_batch(to_cpu_f32(pos.detach(), "pos"))
        g = _image_batch(to_cpu_f32(grad_out, "grad_out"), "grad_out")

        g_color = [] if grad_mask[0] else None
        g_pos = [] if grad_mask[1] else None
        with vctx.lock:
            # Advance the stochastic-backward jitter each backward pass
            eng.iter_seed = getattr(eng, "iter_seed", 0) + 1
            for n in range(n_batch):
                _antialias_uploads(eng, md, color_cpu, rast_cpu, pos_b, n,
                                   eng.iter_seed)
                eng.upload("__seed__", as_hwc(g[n]))
                eng.run()
                if g_color is not None:
                    g_color.append(torch.from_numpy(eng.read_grad("img")))
                if g_pos is not None:
                    g_clip_vk = torch.from_numpy(eng.read_grad("clip"))[0]
                    g_pos.append(clip_grad_vk_to_gl(g_clip_vk) * boost)

        out_gc = None
        if g_color is not None:
            out_gc = torch.stack(g_color, dim=0)  # [N, H, W, C]
            if color.dim() == 3:
                out_gc = out_gc[0]
            out_gc = out_gc.to(color.device)
        out_gp = None
        if g_pos is not None:
            out_gp = torch.stack(g_pos, dim=0)  # [N, V, 4]
            if pos.dim() == 2:
                out_gp = out_gp.sum(dim=0)  # pos shared across the batch
            out_gp = out_gp.to(pos.device)
        return (out_gc, None, out_gp, None, None, None, None, None)


def antialias(color, rast, pos, tri, topology_hash=None,
              pos_gradient_boost=1.0, n_samples=8):
    """Drop-in nvdiffrast antialias (Dr.Hair screen-space AA, Eq.3-5).

    Differentiable w.r.t. `color` and `pos`. `n_samples` selects the
    vertex-gradient backward: 0 = exact bbox scan, > 0 = stochastic
    (n jittered edge samples per face per backward, the fast default).
    `topology_hash` is accepted and ignored.
    """
    del topology_hash
    rinfo: RastInfo = lookup(rast, "rasterize")
    vctx = rinfo.extra["ctx"]
    md = vctx.mesh(tri)
    grad_mask = (_needs_grad(color), _needs_grad(pos))
    out = _AntiAliasFn.apply(color, rast, pos, vctx, md, grad_mask,
                             n_samples, pos_gradient_boost)
    return out.to(color.device)


# ------------------------------ rasterize_soft -----------------------------


def _build_rasterize_soft_engine(ctx, md, h, w, radius_px, peels,
                                 train) -> Engine:
    eng = Engine(ctx._ctx)
    clip_l = eng.leaf("clip", _C.VType.VEC4, md.n_verts, 1)
    face_id_l = eng.leaf("face_id", _C.VType.FLOAT, md.n_faces * 3, 1)
    faces_soft_l = eng.leaf("faces_soft", _C.VType.IVEC3, md.n_faces, 1)
    faces_l = eng.leaf("faces_f", _C.VType.VEC3, md.n_faces, 1)
    vft_l = eng.leaf("vft", _C.VType.FLOAT, md.n_verts, md.max_deg)

    soft_clip = f.soft_clip(clip_l, faces_l, w, h, radius_px)  # GPU unweld
    zero, one = f.constant(0.0), f.constant(1.0)
    outs = []
    prev_shift = None
    for peel in range(peels):
        if peel == 0:
            rs = f.rasterize_soft(soft_clip, face_id_l, faces_soft_l, clip_l,
                                  faces_l, vft_l, w, h, radius_px)
        else:
            rs = f.rasterize_soft_peel(soft_clip, face_id_l, faces_soft_l,
                                       clip_l, faces_l, vft_l, prev_shift, w,
                                       h, radius_px)
        outs.append(rs)
        if peel + 1 < peels:
            # Next peel's threshold: this layer's Eq.3 shifted depth
            # rebuilt from the output channels (0 at background)
            dist_c, cov = f.get_x(rs), f.get_w(rs)
            ge = f.step(zero, dist_c)
            prev_shift = cov * (
                ge * (f.clamp(f.get_z(rs), zero, one) * 0.5)
                + (1.0 - ge) * (f.clamp(dist_c * (-1.0 / radius_px), zero,
                                        one) * 0.5 + 0.5))
    eng.finalize(outs, grad_leaves=["clip"] if train else [])
    return eng


def _rasterize_soft_uploads(eng, md, pos_b, n):
    eng.upload("face_id", md.corner_face_id, token=("static", id(md)), ref=md)
    eng.upload("faces_soft", md.faces_seq_f32, token=("seq", id(md)), ref=md)
    eng.upload("faces_f", md.faces_f32, token=("faces", id(md)), ref=md)
    eng.upload("vft", md.vtx_faces_tex, token=("vft", id(md)), ref=md)
    pos_n = pos_b[n if pos_b.shape[0] > 1 else 0]
    eng.upload("clip", vertex_array(clip_gl_to_vk(pos_n)),
               token=("clip", upload_token(pos_b), n), ref=pos_b)


class _RasterizeSoftFn(torch.autograd.Function):
    @staticmethod
    def forward(fctx, pos, vctx, md, resolution, radius_px, peels, train):
        h, w = resolution
        pos_b = split_batch(to_cpu_f32(pos.detach(), "pos"))
        n_batch = pos_b.shape[0]
        key = ("rasterize_soft", h, w, md.n_faces, md.n_verts, radius_px,
               peels, train)
        eng: Engine = vctx.engine(
            key, lambda: _build_rasterize_soft_engine(vctx, md, h, w,
                                                      radius_px, peels,
                                                      train))
        outs = []
        with vctx.lock:
            for n in range(n_batch):
                _rasterize_soft_uploads(eng, md, pos_b, n)
                eng.run()
                outs.append(torch.stack(
                    [torch.from_numpy(eng.read_output(k))
                     for k in range(peels)], dim=0))
        fctx.dressi = (vctx, eng, md, n_batch, peels)
        fctx.save_for_backward(pos)
        return _stack_batch(outs)  # [N, K, H, W, 4]

    @staticmethod
    def backward(fctx, grad_out):
        vctx, eng, md, n_batch, peels = fctx.dressi
        (pos,) = fctx.saved_tensors
        pos_b = split_batch(to_cpu_f32(pos.detach(), "pos"))
        g = to_cpu_f32(grad_out, "grad_out")  # [N, K, H, W, 4]

        g_pos = []
        with vctx.lock:
            for n in range(n_batch):
                _rasterize_soft_uploads(eng, md, pos_b, n)
                for k in range(peels):
                    eng.upload(Engine.seed_name(k), as_hwc(g[n, k]))
                eng.run()
                g_clip_vk = torch.from_numpy(eng.read_grad("clip"))[0]
                g_pos.append(clip_grad_vk_to_gl(g_clip_vk))
        gp = torch.stack(g_pos, dim=0)
        if pos.dim() == 2:
            gp = gp.sum(dim=0)
        return gp.to(pos.device), None, None, None, None, None, None


def rasterize_soft(ctx, pos, tri, resolution, radius_px=3.0, peels=1):
    """HardSoftRas soft rasterization (extension; no nvdiffrast analogue).

    Returns [N, K, H, W, 4] with channels (signed_dist_px, face_id,
    hard_depth, coverage) per depth-peel layer K. Differentiable w.r.t.
    `pos` through the signed-distance channel (exact adjacency-bounded
    gather backward). Compose sigmoid/Eq.6 silhouettes in torch, e.g.
    `cov * torch.sigmoid(dist / sigma)`.
    """
    h, w = check_resolution(resolution)
    if peels < 1:
        raise ValueError("peels must be >= 1")
    md = ctx.mesh(tri)
    train = _needs_grad(pos)
    out = _RasterizeSoftFn.apply(pos, ctx, md, (h, w), float(radius_px),
                                 int(peels), train)
    return out.to(pos.device)
