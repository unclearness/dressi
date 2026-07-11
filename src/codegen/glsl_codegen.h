#ifndef DRESSI_CODEGEN_GLSL_CODEGEN_H
#define DRESSI_CODEGEN_GLSL_CODEGEN_H

#include <string>

#include "pack/substage.h"

namespace dressi {

// The shared pass-through vertex shader rasterizing a fullscreen triangle
// (gl_VertexIndex trick; draw with 3 vertices, no vertex buffer).
std::string FullscreenVertShader();

// Generates the fragment shader for a substage: joins the GLSL snippets of
// its functions with substage-local normalized names (v0..vN), declares the
// classified inputs, and writes the padded color attachment outputs.
// Binding convention (shared with the executor), set=0:
//   [0, n_inp)                       input attachments
//   [n_inp, n_inp+n_tex)             texture samplers (UV sampling)
//   [n_inp+n_tex, ...+n_slt)         sampler-less textures (texelFetch)
//   [...+n_slt, ...+n_uif)           vec4 UBOs ({1,1} leaf uniforms)
// `max_dim` is the device maxImageDimension2D: any {N,1} texelFetch input
// wider than it is stored 2-D-tiled (see PhysImgSize) and addressed by the
// linear-index fold. The default 0 disables tiling, so off-device callers
// (golden tests) get the untiled {N,1} shaders unchanged.
std::string GenerateFragShader(const SubStage& substage, uint32_t max_dim = 0);

// Compute variant for COMP substages (band-pass gathers): all inputs are
// texelFetch samplers / UBOs (no input attachments), outputs are writeonly
// storage images bound after the uif UBOs; one invocation per pixel
// (local_size_x = 64 along the row).
std::string GenerateCompShader(const SubStage& substage, uint32_t max_dim = 0);

// Generates the pass-through vertex shader and the interpolation fragment
// shader for a RASTER substage (one F::Rasterize function).
struct RasterShaders {
    std::string vert;
    std::string frag;
};
RasterShaders GenerateRasterShaders(const SubStage& substage,
                                    uint32_t max_dim = 0);

// Physical channel count of an image for a VType (VEC3 pads to 4)
uint32_t PhysChannels(VType vtype);

}  // namespace dressi

#endif  // DRESSI_CODEGEN_GLSL_CODEGEN_H
