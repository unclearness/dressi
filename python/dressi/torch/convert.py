"""Tensor validation, device policy, and coordinate conversions."""

from __future__ import annotations

import warnings

import numpy as np
import torch

_warned_device = False
_warned_dtype = False


def to_cpu_f32(t: torch.Tensor, name: str) -> torch.Tensor:
    """Device/dtype policy: CPU float32 contiguous, warning once on coercion."""
    global _warned_device, _warned_dtype
    if t.device.type != "cpu":
        if not _warned_device:
            warnings.warn(
                f"dressi.torch: '{name}' is on {t.device}; v1 transfers "
                "through CPU tensors (a copy per call). Pass CPU tensors to "
                "silence this warning.",
                stacklevel=3,
            )
            _warned_device = True
        t = t.cpu()
    if t.dtype != torch.float32:
        if t.dtype == torch.float64 and not _warned_dtype:
            warnings.warn(
                f"dressi.torch: '{name}' is float64; downcasting to "
                "float32 (the engine is fp32).",
                stacklevel=3,
            )
            _warned_dtype = True
        t = t.float()
    return t.contiguous()


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
