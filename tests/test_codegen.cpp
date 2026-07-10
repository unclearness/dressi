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
