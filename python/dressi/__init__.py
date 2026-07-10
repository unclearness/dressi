"""dressi: Vulkan-backed differentiable rasterization (Dressi, EG 2022).

`dressi._C` exposes the native graph API (DressiAD / Variable / f ops);
`dressi.torch` provides the nvdiffrast-compatible PyTorch layer.
"""

from . import _C  # noqa: F401

__version__ = "0.1.0"
