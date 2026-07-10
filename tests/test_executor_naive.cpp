// GPU executor tests (naive/trivial packing): forward parity against the CPU
// evaluator, gradient execution, and an end-to-end optimization loop.

#include <gtest/gtest.h>

#include <cmath>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"

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

}  // namespace

TEST(ExecutorNaive, ForwardParityElementwise) {
    const uint32_t W = 17, H = 9;
    Variable a(VEC3, {W, H});
    Variable b(VEC3, {W, H});
    Variable s(FLOAT, {1, 1});
    Variable out = F::Mix(F::Abs(a - b), F::Exp(a * 0.3f), F::Float(0.4f)) +
                   s * F::Sqrt(F::Abs(b) + 1.f) +
                   F::Pow(F::Abs(a) + 0.5f, F::Float(2.2f));
    // A scalar loss so BuildBackward preconditions hold
    Variable loss = F::Mean(out);

    auto img_a = MakeImage(W, H, 3,
                           [](size_t i) { return 0.2f + 0.01f * float(i % 37); });
    auto img_b = MakeImage(W, H, 3,
                           [](size_t i) { return 0.9f - 0.008f * float(i % 53); });
    CpuImage img_s(1, 1, 1);
    img_s.data[0] = 1.3f;

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    ad.sendImg(a, img_a);
    ad.sendImg(b, img_b);
    ad.sendImg(s, img_s);
    ad.execStep();

    CpuEvaluator ev;
    ev.bindImage(a, img_a);
    ev.bindImage(b, img_b);
    ev.bindImage(s, img_s);

    ExpectImagesNear(ad.recvImg(out), CpuImageFromTensor(ev.eval(out)), 1e-5f);
    ExpectImagesNear(ad.recvImg(loss), CpuImageFromTensor(ev.eval(loss)),
                     1e-4f);
}

TEST(ExecutorNaive, ForwardParityVectorOps) {
    const uint32_t W = 8, H = 8;
    Variable n(VEC3, {W, H});
    Variable l(VEC3, {1, 1});
    Variable shaded = F::Max(F::Dot(F::Normalize(n), F::Normalize(l)),
                             F::Float(0.f));
    Variable loss = F::Mean(shaded);

    auto img_n = MakeImage(W, H, 3, [](size_t i) {
        return 0.1f + 0.05f * float(i % 11);
    });
    CpuImage img_l(1, 1, 3);
    img_l.data = {0.3f, 0.5f, 0.8f};

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    ad.sendImg(n, img_n);
    ad.sendImg(l, img_l);
    ad.execStep();

    CpuEvaluator ev;
    ev.bindImage(n, img_n);
    ev.bindImage(l, img_l);
    ExpectImagesNear(ad.recvImg(shaded), CpuImageFromTensor(ev.eval(shaded)),
                     1e-5f);
}

TEST(ExecutorNaive, ReductionParityOddSizes) {
    const uint32_t W = 37, H = 23;
    Variable img(FLOAT, {W, H});
    Variable loss = F::Mean(img * img);

    auto data = MakeImage(W, H, 1, [](size_t i) {
        return 0.25f * float(i % 17) - 1.f;
    });

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    ad.sendImg(img, data);
    ad.execStep();

    CpuEvaluator ev;
    ev.bindImage(img, data);
    ExpectImagesNear(ad.recvImg(loss), CpuImageFromTensor(ev.eval(loss)),
                     1e-4f);
}

TEST(ExecutorNaive, GradientParityOnGpu) {
    const uint32_t W = 6, H = 5;
    Variable param(VEC3, {W, H});
    Variable target(VEC3, {W, H});
    Variable diff = target - param;
    Variable loss = F::Mean(diff * diff);
    Variable param_mut = param;
    param_mut.setRequiresGradRecursively();

    auto img_p = MakeImage(W, H, 3,
                           [](size_t i) { return 0.1f * float(i % 13); });
    auto img_t = MakeImage(W, H, 3,
                           [](size_t i) { return 1.f - 0.07f * float(i % 11); });

    // Identity "optimizer" exposing the gradient graph for execution
    Variables grad_vars;
    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        grad_vars = gxs;
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * 0.f);
        }
        return updated;
    });
    ad.sendImg(param, img_p);
    ad.sendImg(target, img_t);
    ad.execStep();

    ASSERT_EQ(grad_vars.size(), 1u);
    CpuEvaluator ev;
    ev.bindImage(param, img_p);
    ev.bindImage(target, img_t);
    ExpectImagesNear(ad.recvImg(grad_vars[0]),
                     CpuImageFromTensor(ev.eval(grad_vars[0])), 1e-5f);
}

TEST(ExecutorNaive, ImageFittingConverges) {
    const uint32_t W = 16, H = 16;
    Variable param(VEC3, {W, H});
    Variable target(VEC3, {W, H});
    Variable diff = target - param;
    Variable loss = F::Mean(diff * diff);
    Variable param_mut = param;
    param_mut.setRequiresGradRecursively();

    // MSE mean divides by N; scale the rate to keep per-pixel steps ~0.4
    const float lr = 0.4f * float(W * H * 3) / 2.f;

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    ad.setOptimizer([lr](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * lr);
        }
        return updated;
    });

    auto img_t = MakeImage(W, H, 3, [W](size_t i) {
        const size_t p = i / 3;
        return 0.5f + 0.5f * std::sin(0.3f * float(p % W)) *
                              std::cos(0.2f * float(p / W));
    });
    ad.sendImg(param, MakeImage(W, H, 3, [](size_t) { return 0.5f; }));
    ad.sendImg(target, img_t);

    ad.execStep();
    const float loss0 = ad.recvImg(loss).data[0];
    for (int iter = 0; iter < 60; iter++) {
        ad.execStep();
    }
    const float loss1 = ad.recvImg(loss).data[0];
    EXPECT_LT(loss1, loss0 * 0.05f);

    const CpuImage optimized = ad.recvImg(param);
    float max_err = 0.f;
    for (size_t i = 0; i < optimized.data.size(); i++) {
        max_err = std::max(max_err,
                           std::abs(optimized.data[i] - img_t.data[i]));
    }
    EXPECT_LT(max_err, 0.05f);
}
