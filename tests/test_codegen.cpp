// GLSL codegen golden tests: stable substage-local naming and the exact
// shader text for small packed substages.

#include <gtest/gtest.h>

#include "codegen/glsl_codegen.h"
#include "core/build_backward.h"
#include "pack/packing.h"
#include "dressi/dressi.h"

using namespace dressi;

TEST(Codegen, FullscreenVertShaderIsStable) {
    const std::string vs = FullscreenVertShader();
    EXPECT_NE(vs.find("gl_VertexIndex"), std::string::npos);
    EXPECT_NE(vs.find("gl_Position"), std::string::npos);
}

TEST(Codegen, SingleFunctionSubstage) {
    Variable a(FLOAT, {8, 8});
    Variable b(FLOAT, {8, 8});
    Variable y = F::Add(a, b);

    Functions funcs = TraverseFuncs({y});
    SubStages substages = TrivialPackSubStages(funcs);
    ASSERT_EQ(substages.size(), 1u);
    const std::string code = GenerateFragShader(substages[0]);

    const std::string expected = R"(#version 450
layout(set=0, binding=0) uniform sampler2D u_slt0;
layout(set=0, binding=1) uniform sampler2D u_slt1;
layout(location=0) out float o_0;
void main() {
    ivec2 dressi_coord = ivec2(gl_FragCoord.xy);
    float v0 = texelFetch(u_slt0, dressi_coord, 0).x;
    float v1 = texelFetch(u_slt1, dressi_coord, 0).x;
    float v2; v2=v0+v1;
    o_0 = v2;
}
)";
    EXPECT_EQ(code, expected);
}

TEST(Codegen, FusedSubstageWithConstantsAndPromotion) {
    Variable x(VEC3, {8, 8});
    // pow(vec3, scalar) forces explicit scalar promotion; the 0.5 constant
    // is inlined instead of becoming a texture input
    Variable y = F::Pow(F::Abs(x), F::Float(0.5f)) * 2.f;

    Functions funcs = TraverseFuncs({y});
    SubStages substages =
            PackFunctionsIntoSubStages(funcs, {y}, PackingLimits{});
    ASSERT_EQ(substages.size(), 1u);
    const std::string code = GenerateFragShader(substages[0]);

    EXPECT_NE(code.find(",vec3(0.5));"), std::string::npos) << code;
    EXPECT_NE(code.find("*vec3(2.0);"), std::string::npos) << code;
    // VEC3 output pads to vec4
    EXPECT_NE(code.find("out vec4 o_0"), std::string::npos) << code;
    EXPECT_NE(code.find("o_0 = vec4("), std::string::npos) << code;
    // Constants never become bindings
    EXPECT_EQ(code.find("u_slt1"), std::string::npos) << code;
}

TEST(Codegen, DeterministicAcrossCalls) {
    Variable x(VEC3, {8, 8});
    Variable y = F::Exp(F::Sin(x) * 0.3f);
    Functions funcs = TraverseFuncs({y});
    SubStages s1 = PackFunctionsIntoSubStages(funcs, {y}, PackingLimits{});
    const std::string c1 = GenerateFragShader(s1[0]);
    const std::string c2 = GenerateFragShader(s1[0]);
    EXPECT_EQ(c1, c2);
}

TEST(Tiling, PhysicalSizeFoldsOversized1D) {
    // Under the limit: untiled {N,1} passthrough.
    EXPECT_FALSE(NeedsTiling(ImgSize{16384, 1}, 16384));
    EXPECT_EQ(PhysImgSize(ImgSize{16384, 1}, 16384), (ImgSize{16384, 1}));
    // A wider row folds into a power-of-two-wide tile within the limit.
    EXPECT_TRUE(NeedsTiling(ImgSize{30338, 1}, 16384));
    const ImgSize p = PhysImgSize(ImgSize{30338, 1}, 16384);
    EXPECT_EQ(p.w, 16384u);           // largest power of two <= 16384
    EXPECT_EQ(p.h, 2u);               // ceil(30338 / 16384)
    EXPECT_LE(p.w, 16384u);
    EXPECT_LE(p.h, 16384u);
    EXPECT_GE(size_t(p.w) * p.h, 30338u);
    // Only 1-D rows tile; a 2-D table is left alone (its own extents must fit).
    EXPECT_FALSE(NeedsTiling(ImgSize{30338, 4}, 16384));
    // max_dim == 0 disables tiling (the off-device / golden-test default).
    EXPECT_FALSE(NeedsTiling(ImgSize{30338, 1}, 0));
    // The fold's linear index round-trips: i -> (i & (W-1), i >> log2 W) -> i.
    const uint32_t w = TileWidth(16384);
    for (uint32_t i : {0u, 1u, 16383u, 16384u, 30337u}) {
        EXPECT_EQ((i / w) * w + (i % w), i);
    }
}

TEST(Tiling, CodegenFoldsWideIndexFetch) {
    const ImgSize scr{4, 4};
    Variable img(FLOAT, scr);
    Variable tri(FLOAT, scr);
    Variable vclip(VEC4, {4, 1});       // small -> never tiled
    Variable faces(VEC3, {8, 1});       // wide relative to max_dim below
    Variable vtx_faces(FLOAT, {4, 2});
    Variable seed(FLOAT, {1, 1});
    Variable y = F::AntiAlias(img, tri, vclip, faces, vtx_faces, seed, 0);

    Functions funcs = TraverseFuncs({y});
    SubStages substages = TrivialPackSubStages(funcs);
    std::string tiled, plain;
    for (const auto& ss : substages) {
        if (ss.shader_type == FRAG || ss.shader_type == RASTER) {
            tiled += GenerateFragShader(ss, /*max_dim=*/4);
            plain += GenerateFragShader(ss, /*max_dim=*/0);
        }
    }
    // With max_dim=4, faces {8,1} tiles (W=4): its fetch folds to & 3 / >> 2,
    // while the untiled vclip fetch and the max_dim=0 shader keep ivec2(_, 0).
    EXPECT_NE(tiled.find("texelFetch(faces, ivec2((owner - 1) & 3, "
                         "(owner - 1) >> 2), 0)"),
              std::string::npos)
            << tiled;
    EXPECT_NE(tiled.find("texelFetch(vclip, ivec2(vi[k], 0), 0)"),
              std::string::npos)
            << tiled;
    EXPECT_NE(plain.find("texelFetch(faces, ivec2(owner - 1, 0), 0)"),
              std::string::npos)
            << plain;
    EXPECT_EQ(plain.find(" & 3"), std::string::npos) << plain;
}
