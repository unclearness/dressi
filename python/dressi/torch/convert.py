"""Tensor validation, device policy, and coordinate conversions."""

from __future__ import annotations

import warnings
import weakref

import numpy as np
import torch

_warned_device = False
_warned_dtype = False

# CPU-copy cache for device tensors: id(gpu tensor) -> (weakref, _version,
# cpu copy). Ops both look up incoming tensors here and register their own
# outputs, so a value produced by one op and consumed by the next does not
# cross the device boundary twice.
_cpu_cache: dict[int, tuple] = {}


def _cache_put(gpu_t: torch.Tensor, cpu_t: torch.Tensor):
    key = id(gpu_t)

    def _drop(_ref, _key=key):
        _cpu_cache.pop(_key, None)

    _cpu_cache[key] = (weakref.ref(gpu_t, _drop), gpu_t._version, cpu_t)


def register_cpu_copy(gpu_t: torch.Tensor, cpu_t: torch.Tensor):
    """Ops call this when returning a device tensor whose CPU contents
    they already hold."""
    if gpu_t.device.type != "cpu":
        _cache_put(gpu_t, cpu_t)


def to_cpu_f32(t: torch.Tensor, name: str) -> torch.Tensor:
    """Device/dtype policy: detached CPU float32 contiguous, warning once
    on coercion. Pass the ORIGINAL tensor (not .detach()) so the CPU-copy
    cache can key on its identity."""
    global _warned_device, _warned_dtype
    if t.device.type != "cpu":
        entry = _cpu_cache.get(id(t))
        if (entry is not None and entry[0]() is t
                and entry[1] == t._version):
            return entry[2]
        if not _warned_device:
            warnings.warn(
                f"dressi.torch: '{name}' is on {t.device}; v1 transfers "
                "through CPU tensors (a copy per call). Pass CPU tensors to "
                "silence this warning.",
                stacklevel=3,
            )
            _warned_device = True
        cpu = t.detach().cpu()
        if cpu.dtype != torch.float32:
            cpu = cpu.float()
        cpu = cpu.contiguous()
        _cache_put(t, cpu)
        return cpu
    out = t.detach()
    if out.dtype != torch.float32:
        if out.dtype == torch.float64 and not _warned_dtype:
            warnings.warn(
                f"dressi.torch: '{name}' is float64; downcasting to "
                "float32 (the engine is fp32).",
                stacklevel=3,
            )
            _warned_dtype = True
        out = out.float()
    return out.contiguous()


def as_hwc(t: torch.Tensor) -> np.ndarray:
    """(H, W, C) float32 contiguous numpy view of a CPU tensor (no copy)."""
    assert t.dim() == 3 and t.dtype == torch.float32 and t.device.type == "cpu"
    return t.contiguous().numpy()


def vertex_array(t: torch.Tensor) -> np.ndarray:
    """{V,1} leaf layout: (V, C) tensor -> (1, V, C) ndarray."""
    return as_hwc(t.unsqueeze(0))


def tri_int32(tri: torch.Tensor) -> torch.Tensor:
    if tri.dim() != 2 or tri.shape[1] != 3:
        raise ValueError(f"tri must be [F, 3] int, got {tuple(tri.shape)}")
    if tri.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"tri must be int32/int64, got {tri.dtype}")
    return tri.cpu().to(torch.int32).contiguous()


def clip_gl_to_vk(pos: torch.Tensor) -> torch.Tensor:
    """OpenGL clip z in [-w, w] -> Vulkan clip z in [0, w] (per vertex).

    z_vk = 0.5 * (z_gl + w); x, y, w unchanged.
    """
    out = pos.clone()
    out[..., 2] = 0.5 * (pos[..., 2] + pos[..., 3])
    return out


def clip_grad_vk_to_gl(g_vk: torch.Tensor) -> torch.Tensor:
    """Transpose of clip_gl_to_vk for backward: given dL/d(clip_vk), return
    dL/d(clip_gl). dz_vk/dz_gl = 0.5, dz_vk/dw = 0.5."""
    out = g_vk.clone()
    out[..., 2] = 0.5 * g_vk[..., 2]
    out[..., 3] = g_vk[..., 3] + 0.5 * g_vk[..., 2]
    return out


def check_resolution(resolution) -> tuple[int, int]:
    h, w = int(resolution[0]), int(resolution[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"resolution must be positive [H, W], got {resolution}")
    return h, w


def split_batch(pos: torch.Tensor, name: str = "pos") -> torch.Tensor:
    """Accept [V, C] or [N, V, C]; return [N, V, C]."""
    if pos.dim() == 2:
        return pos.unsqueeze(0)
    if pos.dim() == 3:
        return pos
    raise ValueError(f"{name} must be [V, C] or [N, V, C], got {tuple(pos.shape)}")
