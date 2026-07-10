// CPU evaluator tests: per-op forward values against analytic references,
// including a PBR-shaped composite (GGX + Smith + Fresnel-Schlick).

#include <gtest/gtest.h>

#include <cmath>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"

using namespace dressi;

namespace {

CpuTensor ScalarTensor(float v) {
    CpuTensor t;
    t.vtype = FLOAT;
    t.size = {1, 1};
    t.data = {v};
    return t;
}

CpuTensor Vec3Tensor(float x, float y, float z) {
    CpuTensor t;
    t.vtype = VEC3;
    t.size = {1, 1};
    t.data = {x, y, z};
    return t;
}

float EvalScalar1(Variable (*op)(const Variable&), float x) {
    Variable vx(FLOAT, {1, 1});
    Variable y = op(vx);
    CpuEvaluator ev;
    ev.bind(vx, ScalarTensor(x));
    return ev.eval(y).data[0];
}

float EvalScalar2(Variable (*op)(const Variable&, const Variable&), float a,
                  float b) {
    Variable va(FLOAT, {1, 1});
    Variable vb(FLOAT, {1, 1});
    Variable y = op(va, vb);
    CpuEvaluator ev;
    ev.bind(va, ScalarTensor(a));
    ev.bind(vb, ScalarTensor(b));
    return ev.eval(y).data[0];
}

}  // namespace

TEST(CpuEval, UnaryMath) {
    EXPECT_FLOAT_EQ(EvalScalar1(F::Sin, 0.5f), std::sin(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Cos, 0.5f), std::cos(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Tan, 0.5f), std::tan(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Asin, 0.5f), std::asin(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Acos, 0.5f), std::acos(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Atan, 0.5f), std::atan(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Exp, 0.5f), std::exp(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Exp2, 0.5f), std::exp2(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Log, 0.5f), std::log(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Log2, 0.5f), std::log2(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Sqrt, 0.5f), std::sqrt(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::InverseSqrt, 0.5f), 1.f / std::sqrt(0.5f));
    EXPECT_FLOAT_EQ(EvalScalar1(F::Abs, -0.5f), 0.5f);
    EXPECT_FLOAT_EQ(EvalScalar1(F::Sign, -0.5f), -1.f);
    EXPECT_FLOAT_EQ(EvalScalar1(F::Floor, 1.7f), 1.f);
    EXPECT_FLOAT_EQ(EvalScalar1(F::Fract, 1.7f), 0.7f);
}

TEST(CpuEval, BinaryMath) {
    EXPECT_FLOAT_EQ(EvalScalar2(F::Add, 2.f, 3.f), 5.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Sub, 2.f, 3.f), -1.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Mul, 2.f, 3.f), 6.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Div, 3.f, 2.f), 1.5f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Pow, 2.f, 3.f), 8.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Max, 2.f, 3.f), 3.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Min, 2.f, 3.f), 2.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Mod, 7.f, 3.f), 1.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Atan2, 1.f, 2.f), std::atan2(1.f, 2.f));
    EXPECT_FLOAT_EQ(EvalScalar2(F::Step, 2.f, 3.f), 1.f);
    EXPECT_FLOAT_EQ(EvalScalar2(F::Step, 3.f, 2.f), 0.f);
}

TEST(CpuEval, OperatorsAndConstants) {
    Variable x(FLOAT, {1, 1});
    Variable y = (2.f * x + 1.f) / (x - 4.f);
    CpuEvaluator ev;
    ev.bind(x, ScalarTensor(2.f));
    EXPECT_FLOAT_EQ(ev.eval(y).data[0], (2.f * 2.f + 1.f) / (2.f - 4.f));
}

TEST(CpuEval, VectorOps) {
    Variable a(VEC3, {1, 1});
    Variable b(VEC3, {1, 1});
    CpuEvaluator ev;
    ev.bind(a, Vec3Tensor(1.f, 2.f, 3.f));
    ev.bind(b, Vec3Tensor(4.f, -5.f, 6.f));

    EXPECT_FLOAT_EQ(ev.eval(F::Dot(a, b)).data[0], 1.f * 4 - 2.f * 5 + 3.f * 6);

    auto cross = ev.eval(F::Cross(a, b));
    EXPECT_FLOAT_EQ(cross.data[0], 2.f * 6 - 3.f * (-5));
    EXPECT_FLOAT_EQ(cross.data[1], 3.f * 4 - 1.f * 6);
    EXPECT_FLOAT_EQ(cross.data[2], 1.f * (-5) - 2.f * 4);

    const float len = std::sqrt(1.f + 4.f + 9.f);
    EXPECT_FLOAT_EQ(ev.eval(F::Length(a)).data[0], len);

    auto norm = ev.eval(F::Normalize(a));
    EXPECT_FLOAT_EQ(norm.data[1], 2.f / len);

    // reflect(I,N) with N normalized
    Variable n = F::Normalize(b);
    auto refl = ev.eval(F::Reflect(a, n));
    // reference: I - 2*dot(N,I)*N
    auto nn = ev.eval(n);
    const float d = nn.data[0] * 1.f + nn.data[1] * 2.f + nn.data[2] * 3.f;
    EXPECT_NEAR(refl.data[0], 1.f - 2.f * d * nn.data[0], 1e-5f);
    EXPECT_NEAR(refl.data[2], 3.f - 2.f * d * nn.data[2], 1e-5f);

    EXPECT_FLOAT_EQ(ev.eval(F::CompSum(a)).data[0], 6.f);
    EXPECT_FLOAT_EQ(ev.eval(F::GetY(a)).data[0], 2.f);

    auto v4 = ev.eval(F::Vec4(a, F::GetX(b)));
    EXPECT_FLOAT_EQ(v4.data[3], 4.f);
}

TEST(CpuEval, ScalarVectorBroadcast) {
    Variable s(FLOAT, {1, 1});
    Variable v(VEC3, {1, 1});
    CpuEvaluator ev;
    ev.bind(s, ScalarTensor(2.f));
    ev.bind(v, Vec3Tensor(1.f, 2.f, 3.f));
    auto y = ev.eval(s * v);
    EXPECT_EQ(y.vtype, VEC3);
    EXPECT_FLOAT_EQ(y.data[2], 6.f);
}

TEST(CpuEval, MatrixOps) {
    Variable m(MAT3, {1, 1});
    Variable v(VEC3, {1, 1});
    CpuEvaluator ev;
    // Column-major: columns (1,0,0), (0,2,0), (0,0,4) with m[2][1]=3
    // i.e. col2 = (0,3,4)
    CpuTensor mt;
    mt.vtype = MAT3;
    mt.size = {1, 1};
    mt.data = {1, 0, 0, 0, 2, 0, 0, 3, 4};
    ev.bind(m, mt);
    ev.bind(v, Vec3Tensor(1.f, 1.f, 1.f));

    auto mv = ev.eval(F::MatMul(m, v));
    EXPECT_FLOAT_EQ(mv.data[0], 1.f);
    EXPECT_FLOAT_EQ(mv.data[1], 2.f + 3.f);
    EXPECT_FLOAT_EQ(mv.data[2], 4.f);

    auto mtj = ev.eval(F::Transpose(m));
    EXPECT_FLOAT_EQ(mtj.data[1 * 3 + 2], 3.f);  // transposed position

    auto outer = ev.eval(F::OuterProduct(v, v));
    EXPECT_EQ(outer.vtype, MAT3);
    EXPECT_FLOAT_EQ(outer.data[0], 1.f);
}

TEST(CpuEval, SpatialImageOpsAndReductions) {
    const uint32_t W = 37, H = 23;
    Variable img(FLOAT, {W, H});
    CpuTensor t;
    t.vtype = FLOAT;
    t.size = {W, H};
    t.data.resize(size_t(W) * H);
    double ref_sum = 0.0;
    for (size_t i = 0; i < t.data.size(); i++) {
        t.data[i] = float(i % 17) * 0.25f - 1.f;
        ref_sum += t.data[i];
    }
    CpuEvaluator ev;
    ev.bind(img, t);

    auto sum = ev.eval(F::Sum(img));
    EXPECT_EQ(sum.size, (ImgSize{1, 1}));
    EXPECT_NEAR(sum.data[0], float(ref_sum), 1e-2f);

    auto mean = ev.eval(F::Mean(img));
    EXPECT_NEAR(mean.data[0], float(ref_sum / (W * H)), 1e-5f);
}

TEST(CpuEval, SumPixelWise) {
    Variable a(FLOAT, {4, 4});
    Variable b(FLOAT, {4, 4});
    Variable c(FLOAT, {1, 1});
    Variable y = F::SumPixelWise({a, b, c});
    EXPECT_EQ(y.getImgSize(), (ImgSize{4, 4}));

    CpuEvaluator ev;
    CpuTensor ta, tb;
    ta.vtype = tb.vtype = FLOAT;
    ta.size = tb.size = {4, 4};
    ta.data.assign(16, 1.f);
    tb.data.assign(16, 2.f);
    ev.bind(a, ta);
    ev.bind(b, tb);
    ev.bind(c, ScalarTensor(10.f));
    EXPECT_FLOAT_EQ(ev.eval(y).data[5], 13.f);
}

// PBR-shaped composite: Cook-Torrance GGX specular for a single direction
// setup, evaluated per-pixel over roughness values.
TEST(CpuEval, PbrCompositeExpressible) {
    const uint32_t W = 8, H = 8;
    Variable n(VEC3, {W, H});      // normal
    Variable v(VEC3, {1, 1});      // view dir
    Variable l(VEC3, {1, 1});      // light dir
    Variable rough(FLOAT, {W, H});
    Variable f0(VEC3, {1, 1});

    Variable h = F::Normalize(v + l);
    Variable n_dot_h = F::Max(F::Dot(n, h), F::Float(0.f));
    Variable n_dot_v = F::Max(F::Dot(n, v), F::Float(1e-4f));
    Variable n_dot_l = F::Max(F::Dot(n, l), F::Float(1e-4f));
    Variable v_dot_h = F::Max(F::Dot(v, h), F::Float(0.f));

    // GGX normal distribution
    Variable alpha = rough * rough;
    Variable alpha2 = alpha * alpha;
    Variable denom_d = n_dot_h * n_dot_h * (alpha2 - 1.f) + 1.f;
    Variable dist =
            alpha2 / (F::Float(3.14159265f) * denom_d * denom_d);

    // Smith GGX geometry (Schlick-GGX)
    Variable k = (rough + 1.f) * (rough + 1.f) / 8.f;
    Variable g_v = n_dot_v / (n_dot_v * (1.f - k) + k);
    Variable g_l = n_dot_l / (n_dot_l * (1.f - k) + k);
    Variable geom = g_v * g_l;

    // Fresnel-Schlick
    Variable fresnel =
            f0 + (1.f - f0) * F::Pow(1.f - v_dot_h, F::Float(5.f));

    Variable spec = fresnel * (dist * geom /
                               (4.f * n_dot_v * n_dot_l + 1e-4f));
    // Tone map (Reinhard) + gamma
    Variable mapped = spec / (spec + 1.f);
    Variable ldr = F::Pow(mapped, F::Float(1.f / 2.2f));

    EXPECT_EQ(ldr.getVType(), VEC3);
    EXPECT_EQ(ldr.getImgSize(), (ImgSize{W, H}));

    CpuEvaluator ev;
    CpuTensor tn, tr;
    tn.vtype = VEC3;
    tn.size = {W, H};
    tn.data.resize(size_t(W) * H * 3);
    for (size_t p = 0; p < size_t(W) * H; p++) {
        tn.data[p * 3 + 0] = 0.f;
        tn.data[p * 3 + 1] = 0.f;
        tn.data[p * 3 + 2] = 1.f;
    }
    tr.vtype = FLOAT;
    tr.size = {W, H};
    tr.data.resize(size_t(W) * H);
    for (size_t p = 0; p < tr.data.size(); p++) {
        tr.data[p] = 0.1f + 0.8f * float(p) / float(tr.data.size());
    }
    ev.bind(n, tn);
    ev.bind(rough, tr);
    ev.bind(v, Vec3Tensor(0.f, 0.f, 1.f));
    ev.bind(l, Vec3Tensor(0.f, 0.f, 1.f));
    ev.bind(f0, Vec3Tensor(0.04f, 0.04f, 0.04f));

    auto out = ev.eval(ldr);
    ASSERT_EQ(out.data.size(), size_t(W) * H * 3);
    for (float c : out.data) {
        EXPECT_TRUE(std::isfinite(c));
        EXPECT_GE(c, 0.f);
        EXPECT_LE(c, 1.f);
    }
    // Rougher pixels produce weaker mirror-direction specular
    EXPECT_GT(out.data[0], out.data[(size_t(W) * H - 1) * 3]);
}
