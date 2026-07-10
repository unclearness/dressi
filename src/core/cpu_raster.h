#ifndef DRESSI_CORE_CPU_RASTER_H
#define DRESSI_CORE_CPU_RASTER_H

#include "core/node.h"

namespace dressi {

// Screen-space helpers shared by the CPU rasterizers, the HardSoftRas
// distance kernels and the AA kernels. Conventions (matching the Vulkan
// executor): NDC y is down after PerspectiveVk's y-flip, so
// screen = (clip.xy / clip.w * 0.5 + 0.5) * (W, H) with row 0 at the top;
// pixel centers at +0.5; depth = clip.z / clip.w in [0, 1]; LessOrEqual
// depth test against a clear value of 1; background = 0.

// Point-to-triangle signed distance in screen space (pixels).
// dist >= 0 inside (winding-agnostic), < 0 outside. `edge` is the argmin
// edge k = (s[k], s[(k+1)%3]) and `t` the closest-point parameter on it.
struct TriDist {
    float dist = 0.f;
    int edge = 0;
    float t = 0.f;
};
TriDist SignedDistToTri(const float p[2], const float s[3][2]);

// Per-vertex screen-space gradient of SignedDistToTri (envelope theorem:
// the argmin edge and clamp state are held fixed). g_s[i] accumulates
// d dist / d s[i]; entries for vertices not on the argmin edge are zero.
void SignedDistGradToTri(const float p[2], const float s[3][2],
                         const TriDist& td, float g_s[3][2]);

// 2D screen-space barycentric coordinates of p in triangle s (unclamped).
// Returns false for degenerate triangles.
bool ScreenBarycentric(const float p[2], const float s[3][2], float lambda[3]);

// CPU reference rasterizer for F::Rasterize: depth-tested, perspective-
// correct attribute interpolation. Faces with any |w| <= eps are skipped
// (no near-plane clipping).
CpuTensor RasterizeHardCpu(const CpuTensor& clip, const CpuTensor& attrib,
                           const CpuTensor& faces, ImgSize screen);

// CPU reference for F::RasterizeFaceId: float(face_index) + 1, background 0.
CpuTensor RasterizeFaceIdCpu(const CpuTensor& clip, const CpuTensor& faces,
                             ImgSize screen);

}  // namespace dressi

#endif  // DRESSI_CORE_CPU_RASTER_H
