"""RasterizeVulkanContext: shared Vulkan device + engine/mesh caches."""

from __future__ import annotations

import threading

import torch

from .. import _C
from .meshdata import MeshData, mesh_key


class RasterizeVulkanContext:
    """Vulkan counterpart of nvdiffrast's RasterizeGLContext.

    Owns one headless Vulkan device shared by every cached engine. Engines
    (compiled Dressi graphs) are keyed by shape signature; the first call
    with a new signature pays graph build + shader compile, subsequent
    calls only upload/execute/download. `clear_cache()` frees them.
    """

    def __init__(self, device=None, debug: bool = False):
        # `device` is accepted for nvdiffrast signature compatibility; the
        # Vulkan device is always the system's first physical device in v1.
        del device
        self._ctx = _C.DressiAD.create_context(debug)
        self._engines: dict[tuple, object] = {}
        self._meshes: dict[tuple, MeshData] = {}
        self.lock = threading.RLock()

    def engine(self, key: tuple, builder):
        """Engine for `key`, building it with `builder()` on first use."""
        with self.lock:
            eng = self._engines.get(key)
            if eng is None:
                eng = builder()
                self._engines[key] = eng
            return eng

    def mesh(self, tri: torch.Tensor) -> MeshData:
        key = mesh_key(tri)
        with self.lock:
            md = self._meshes.get(key)
            if md is None:
                md = MeshData(tri)
                self._meshes[key] = md
            return md

    def clear_cache(self):
        with self.lock:
            self._engines.clear()
            self._meshes.clear()
