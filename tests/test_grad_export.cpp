// Gradient-export API tests: setGradOutputsEnabled/inputGrads must expose
// raw leaf gradients without an optimizer, matching the optimizer-lambda
// capture path and the surrogate-seed VJP; plus shared-VkContext reuse
// across DressiAD instances.

#include <gtest/gtest.h>

#include <cmath>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"
#include "grad_capture.h"

using namespace dressi;

namespace {

CpuImage MakeImage(uint32_t w, uint32_t h, uint32_t c,
                   const std::function<float(size_t)>& gen) {
    CpuImage img(w, h, c);
    for (size_t i = 0; i < img.data.size(); i++) {
        img.data[i] = gen(i);
    }
    return img;
}

void ExpectImagesNear(const CpuImage& a, const CpuImage& b, float tol) {
    ASSERT_EQ(a.width, b.width);
    ASSERT_EQ(a.height, b.height);
    ASSERT_EQ(a.channels, b.channels);
    for (size_t i = 0; i < a.data.size(); i++) {
        ASSERT_NEAR(a.data[i], b.data[i], tol) << "index " << i;
    }
}

Variable GradVarOf(const DressiAD& ad, const Variable& wrt) {
    for (const auto& [x, g] : ad.inputGrads()) {
        if (x == wrt) {
            return g;
        }
    }
    return Variable(nullptr);
}

}  // namespace

// loss = (x*x) * seed with a non-uniform, non-requires-grad seed image:
// dL/dx = 2*x*seed. This is the external-VJP injection pattern (the seed
// plays grad_output); the exported gradient must equal the analytic value.
TEST(GradExport, SurrogateSeedVjp) {
    const uint32_t W = 13, H = 7;
    Variable x(FLOAT, {W, H});
    Variable seed(FLOAT, {W, H});
    Variable loss = x * x * seed;
    x.setRequiresGradRecursively(true);

    auto img_x = MakeImage(W, H, 1,
                           [](size_t i) { return 0.3f + 0.02f * float(i % 29); });
    auto img_seed = MakeImage(W, H, 1, [](size_t i) {
        return -1.f + 0.05f * float(i % 41);
    });

    DressiAD ad;
    ad.setLossVar(loss);
    ad.setGradOutputsEnabled(true);
    ad.sendImg(x, img_x);
    ad.sendImg(seed, img_seed);
    ad.execStep();

    Variable gx = GradVarOf(ad, x);
    ASSERT_TRUE(bool(gx));
    CpuImage expected(W, H, 1);
    for (size_t i = 0; i < expected.data.size(); i++) {
        expected.data[i] = 2.f * img_x.data[i] * img_seed.data[i];
    }
    ExpectImagesNear(ad.recvImg(gx), expected, 1e-4f);
}

// The exported gradients must match the identity-optimizer capture path
// (the pre-existing way of reading gradients) on an identical graph.
TEST(GradExport, MatchesOptimizerCapture) {
    const uint32_t W = 9, H = 6;
    const auto build = [&](Variable& x, Variable& t) {
        x = Variable(VEC3, {W, H});
        t = Variable(VEC3, {W, H});
        Variable diff = x - F::StopGradient(t);
        Variable loss = F::Mean(diff * diff);
        x.setRequiresGradRecursively(true);
        return loss;
    };
    auto img_x = MakeImage(W, H, 3,
                           [](size_t i) { return 0.1f + 0.01f * float(i % 17); });
    auto img_t = MakeImage(W, H, 3,
                           [](size_t i) { return 0.8f - 0.02f * float(i % 23); });

    // Path A: grad export, no optimizer
    Variable xa, ta;
    Variable loss_a = build(xa, ta);
    DressiAD ad_a;
    ad_a.setLossVar(loss_a);
    ad_a.setGradOutputsEnabled(true);
    ad_a.sendImg(xa, img_x);
    ad_a.sendImg(ta, img_t);
    ad_a.execStep();
    Variable gx_a = GradVarOf(ad_a, xa);
    ASSERT_TRUE(bool(gx_a));

    // Path B: identity optimizer capturing gxs
    Variable xb, tb;
    Variable loss_b = build(xb, tb);
    DressiAD ad_b;
    ad_b.setLossVar(loss_b);
    dressi_test::GradCapture cap;
    ad_b.setOptimizer(dressi_test::IdentityOptimizer(ad_b, cap));
    ad_b.sendImg(xb, img_x);
    ad_b.sendImg(tb, img_t);
    ad_b.execStep();
    Variable gx_b = dressi_test::GradOf(cap, xb);
    ASSERT_TRUE(bool(gx_b));

    ExpectImagesNear(ad_a.recvImg(gx_a), ad_b.recvImg(gx_b), 1e-5f);
}

// Repeated execSteps with fresh seed uploads (the eager fwd/bwd cadence,
// pruning disabled) must keep returning up-to-date gradients.
TEST(GradExport, SeedUpdatePropagates) {
    const uint32_t W = 5, H = 4;
    Variable x(FLOAT, {W, H});
    Variable seed(FLOAT, {W, H});
    Variable loss = F::Sin(x) * seed;
    x.setRequiresGradRecursively(true);

    auto img_x = MakeImage(W, H, 1,
                           [](size_t i) { return 0.05f * float(i); });
    DressiAD ad;
    ad.setRebuildCounts(0, 0);
    ad.setLossVar(loss);
    ad.setGradOutputsEnabled(true);
    ad.sendImg(x, img_x);

    for (int iter = 0; iter < 3; iter++) {
        auto img_seed = MakeImage(W, H, 1, [&](size_t i) {
            return 0.5f * float(iter + 1) - 0.03f * float(i);
        });
        ad.sendImg(seed, img_seed);
        ad.execStep();
        Variable gx = GradVarOf(ad, x);
        ASSERT_TRUE(bool(gx));
        CpuImage expected(W, H, 1);
        for (size_t i = 0; i < expected.data.size(); i++) {
            expected.data[i] = std::cos(img_x.data[i]) * img_seed.data[i];
        }
        ExpectImagesNear(ad.recvImg(gx), expected, 1e-4f);
    }
}

// GPU parity of the FaceFetch EXACT backward (n_samples == 0) against the
// CPU evaluator: the per-face bbox scan shader must match the CPU kernel.
TEST(GradExport, FaceFetchExactBwdGpuParity) {
    const uint32_t W = 8, H = 6;
    Variable attr(FLOAT, {2, 1});
    Variable idx(FLOAT, {W, H});
    Variable p0(VEC4, {2, 1});
    Variable p1(VEC4, {2, 1});
    Variable p2(VEC4, {2, 1});
    Variable seed(FLOAT, {1, 1});
    Variable wimg(FLOAT, {W, H});
    Variable out = F::FaceFetch(attr, idx, p0, p1, p2, seed, 1.f,
                                /*n_samples=*/0);
    Variable loss = F::Mean(out * wimg);
    attr.setRequiresGradRecursively(true);

    const auto clip_of = [&](float sx, float sy, int c) {
        const float v[4] = {sx / float(W) * 2.f - 1.f,
                            sy / float(H) * 2.f - 1.f, 0.5f, 1.f};
        return v[c];
    };
    CpuImage img_p0(2, 1, 4), img_p1(2, 1, 4), img_p2(2, 1, 4);
    const float c0[2][2] = {{1.f, 1.f}, {3.f, 2.f}};
    const float c1[2][2] = {{5.f, 1.f}, {7.f, 2.f}};
    const float c2[2][2] = {{1.f, 4.f}, {7.f, 5.f}};
    for (uint32_t f = 0; f < 2; f++) {
        for (uint32_t c = 0; c < 4; c++) {
            img_p0.at(f, 0, c) = clip_of(c0[f][0], c0[f][1], int(c));
            img_p1.at(f, 0, c) = clip_of(c1[f][0], c1[f][1], int(c));
            img_p2.at(f, 0, c) = clip_of(c2[f][0], c2[f][1], int(c));
        }
    }
    CpuImage img_idx(W, H, 1, 0.f);
    img_idx.at(2, 1, 0) = 1.f;
    img_idx.at(3, 2, 0) = 1.f;
    img_idx.at(5, 0, 0) = 1.f;
    img_idx.at(4, 3, 0) = 2.f;
    img_idx.at(6, 2, 0) = 2.f;
    img_idx.at(7, 5, 0) = 2.f;
    CpuImage img_attr = MakeImage(2, 1, 1,
                                  [](size_t i) { return 0.4f + 0.3f * float(i); });
    CpuImage img_w = MakeImage(W, H, 1, [](size_t i) {
        return 0.1f + 0.05f * float(i % 13);
    });
    CpuImage img_seed(1, 1, 1, 0.f);

    DressiAD ad;
    ad.setLossVar(loss);
    ad.setGradOutputsEnabled(true);
    ad.sendImg(attr, img_attr);
    ad.sendImg(idx, img_idx);
    ad.sendImg(p0, img_p0);
    ad.sendImg(p1, img_p1);
    ad.sendImg(p2, img_p2);
    ad.sendImg(seed, img_seed);
    ad.sendImg(wimg, img_w);
    ad.execStep();

    Variable gx = GradVarOf(ad, attr);
    ASSERT_TRUE(bool(gx));

    CpuEvaluator ev;
    ev.bindImage(attr, img_attr);
    ev.bindImage(idx, img_idx);
    ev.bindImage(p0, img_p0);
    ev.bindImage(p1, img_p1);
    ev.bindImage(p2, img_p2);
    ev.bindImage(seed, img_seed);
    ev.bindImage(wimg, img_w);
    ExpectImagesNear(ad.recvImg(gx), CpuImageFromTensor(ev.eval(gx)), 1e-5f);
}

// Two DressiAD instances sharing one VkContext must both execute correctly
// (the engine-cache pattern of the Python bindings).
TEST(GradExport, SharedContext) {
    VkContextPtr ctx = DressiAD::createContext();
    ASSERT_TRUE(bool(ctx));

    const uint32_t W = 6, H = 3;
    auto img = MakeImage(W, H, 1,
                         [](size_t i) { return 0.1f * float(i % 7); });

    Variable x1(FLOAT, {W, H});
    Variable loss1 = x1 * 2.f;
    DressiAD ad1(ctx);
    ad1.setLossVar(loss1);
    ad1.markOutput(loss1);
    ad1.sendImg(x1, img);
    ad1.execStep();

    Variable x2(FLOAT, {W, H});
    Variable loss2 = x2 + 1.f;
    DressiAD ad2(ctx);
    ad2.setLossVar(loss2);
    ad2.markOutput(loss2);
    ad2.sendImg(x2, img);
    ad2.execStep();

    CpuImage out1 = ad1.recvImg(loss1);
    CpuImage out2 = ad2.recvImg(loss2);
    for (size_t i = 0; i < out1.data.size(); i++) {
        ASSERT_NEAR(out1.data[i], img.data[i] * 2.f, 1e-5f);
        ASSERT_NEAR(out2.data[i], img.data[i] + 1.f, 1e-5f);
    }
}
