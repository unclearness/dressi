#ifndef DRESSI_EXAMPLES_PBR_GRAPH_H
#define DRESSI_EXAMPLES_PBR_GRAPH_H

// The one Cook-Torrance + split-sum IBL forward graph shared by the
// real-time viewer and the material / envmap optimization examples: the
// optimizers differentiate exactly the shader the viewer displays.
// G-buffer construction and shading are split so an optimization view can
// rasterize once and shade twice (GT materials vs optimized materials).

#include <vector>

#include "dressi/dressi.h"

namespace dressi_examples {

// Static geometry leaves (upload once; `clip` is re-sent per frame/view)
struct PbrGeometryLeaves {
    dressi::Variable clip{nullptr};     // VEC4 {V,1} clip positions
    dressi::Variable uvh{nullptr};      // VEC4 {V,1} (u, v, 1, tangent_w)
    dressi::Variable normal{nullptr};   // VEC3 {V,1} world normals
    dressi::Variable tangent{nullptr};  // VEC3 {V,1} world tangents
    dressi::Variable wpos{nullptr};     // VEC3 {V,1} world positions
    dressi::Variable faces{nullptr};    // IVEC3 {F,1}
};

// Material texture leaves (all VEC3, linear-space color)
struct PbrMaterialLeaves {
    dressi::Variable albedo{nullptr};      // linear base color
    dressi::Variable mr{nullptr};          // G = roughness, B = metallic
    dressi::Variable normal_map{nullptr};  // tangent space
    dressi::Variable ao{nullptr};          // R = AO
    dressi::Variable emissive{nullptr};    // linear
};

// Per-frame/per-view {1,1} uniforms (send the SAME set every frame so the
// reactive cache sees a stable dirty pattern)
struct PbrFrameLeaves {
    dressi::Variable cam_pos{nullptr};    // VEC3 world eye
    dressi::Variable cam_fwd{nullptr};    // VEC3 unit forward
    dressi::Variable cam_right{nullptr};  // VEC3 right * tan(fov/2) * aspect
    dressi::Variable cam_up{nullptr};     // VEC3 up * tan(fov/2)
    dressi::Variable light_pos{nullptr};  // VEC3 world light position
    dressi::Variable light_col{nullptr};  // VEC3 radiant intensity
};

// IBL precompute wired off the env leaf: static once env is uploaded, so
// the reactive cache runs it once and prunes it from the steady frame
struct PbrIblMaps {
    dressi::Variable env{nullptr};  // VEC3 equirect HDR leaf (linear)
    dressi::Variable irr{nullptr};
    std::vector<dressi::Variable> pref;
    dressi::Variable lut{nullptr};
};

// `differentiable_prefilter` swaps the importance-sampled prefilter for
// the deterministic F::PrefilterConv chain (exact transpose backward) —
// required when the env itself is optimized; both branches of an
// optimization must use the SAME variant or the formulation mismatch is
// pushed into the recovered env.
PbrIblMaps BuildPbrIblMaps(const dressi::Variable& env,
                           bool differentiable_prefilter = false);

// Screen-space G-buffer (4 raster passes over one view's clip positions)
struct PbrGBuffer {
    dressi::Variable uv{nullptr};      // VEC2
    dressi::Variable mask{nullptr};    // FLOAT coverage
    dressi::Variable handed{nullptr};  // FLOAT +-1 tangent handedness
    dressi::Variable g_n{nullptr};     // VEC3 interpolated normal (unnorm.)
    dressi::Variable g_t{nullptr};     // VEC3 interpolated tangent (unnorm.)
    dressi::Variable g_p{nullptr};     // VEC3 world position
};

PbrGBuffer BuildPbrGBuffer(const PbrGeometryLeaves& geom,
                           dressi::ImgSize screen);

struct PbrOutputs {
    dressi::Variable ldr{nullptr};     // VEC3 screen: Reinhard + gamma
    dressi::Variable mapped{nullptr};  // VEC3 screen: Reinhard, pre-gamma
                                       // (optimization losses use this —
                                       // the gamma Pow has an infinite
                                       // gradient at 0)
    dressi::Variable hdr{nullptr};     // VEC3 screen: pre-tonemap composite
    dressi::Variable mask{nullptr};    // FLOAT screen coverage
    dressi::Variable uv{nullptr};      // VEC2 screen G-buffer UV
    dressi::Variable albedo{nullptr};  // VEC3 screen sampled albedo (debug)
    dressi::Variable normal{nullptr};  // VEC3 screen shading normal (debug)
};

// Shades one G-buffer. `inv_uv` is the texture-sized VEC4 inverse-UV table
// consumed by the differentiable texture fetches' backward (rasterize the
// mesh in UV space per view); pass a never-uploaded dummy Variable when no
// gradients are needed.
PbrOutputs BuildPbrShade(const PbrGBuffer& gbuf,
                         const PbrMaterialLeaves& mat,
                         const PbrFrameLeaves& frame, const PbrIblMaps& ibl,
                         const dressi::Variable& inv_uv,
                         dressi::ImgSize screen, bool normal_mapping = true);

// Convenience wrapper: G-buffer + shade (the viewer path)
PbrOutputs BuildPbrForward(const PbrGeometryLeaves& geom,
                           const PbrMaterialLeaves& mat,
                           const PbrFrameLeaves& frame,
                           const PbrIblMaps& ibl,
                           const dressi::Variable& inv_uv,
                           dressi::ImgSize screen,
                           bool normal_mapping = true);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_PBR_GRAPH_H
