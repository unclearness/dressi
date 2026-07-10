"""nvdiffrast-compatible PyTorch API backed by the Dressi Vulkan engine.

    import dressi.torch as dr

    ctx = dr.RasterizeVulkanContext()
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[H, W])
    col, _  = dr.interpolate(attr, rast, tri)
    col     = dr.texture(tex, uv)
    col     = dr.antialias(col, rast, pos, tri)

Coordinate conventions follow nvdiffrast (OpenGL clip space, z in [-w, w];
the Vulkan depth remap happens internally). See the project README for the
v1 limitations (CPU tensor transfer, nearest texture filtering, ...).
"""

from .context import RasterizeVulkanContext
from .ops import antialias, interpolate, rasterize, rasterize_soft, texture

__all__ = [
    "RasterizeVulkanContext",
    "rasterize",
    "interpolate",
    "texture",
    "antialias",
    "rasterize_soft",
]
