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
// classified inputs (slt -> sampler2D texelFetch), and writes the padded
// color attachment outputs.
// Binding convention: slt_vars occupy set=0, bindings [0, slt_count).
std::string GenerateFragShader(const SubStage& substage);

// Physical channel count of an image for a VType (VEC3 pads to 4)
uint32_t PhysChannels(VType vtype);

}  // namespace dressi

#endif  // DRESSI_CODEGEN_GLSL_CODEGEN_H
