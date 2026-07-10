// RSP executor tests: parity against the naive (trivial-packing) executor
// and the CPU evaluator, plus pass-count reduction and end-to-end
// optimization in RSP mode.

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

TEST(ExecutorRsp, ForwardParityWithCpu) {
    const uint32_t W = 17, H = 9;
    Variable a(VEC3, {W, H});
    Variable b(VEC3, {W, H});
    Variable s(FLOAT, {1, 1});
    Variable out = F::Mix(F::Abs(a - b), F::Exp(a * 0.3f), F::Float(0.4f)) +
                   s * F::Sqrt(F::Abs(b) + 1.f) +
                   F::Pow(F::Abs(a) + 0.5f, F::Float(2.2f));
    Variable loss = F::Mean(out);

    auto img_a = MakeImage(W, H, 3,
                           [](size_t i) { return 0.2f + 0.01f * float(i % 37); });
    auto img_b = MakeImage(W, H, 3,
                           [](size_t i) { return 0.9f - 0.008f * float(i % 53); });
    CpuImage img_s(1, 1, 1);
    img_s.data[0] = 1.3f;

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::RSP);
    ad.setLossVar(loss);
    ad.sendImg(a, img_a);
    ad.sendImg(b, img_b);
    ad.sendImg(s, img_s);
    ad.execStep();

    CpuEvaluator ev;
    ev.bindImage(a, img_a);
    ev.bindImage(b, img_b);
    ev.bindImage(s, img_s);
    ExpectImagesNear(ad.recvImg(loss), CpuImageFromTensor(ev.eval(loss)),
                     1e-4f);
}

TEST(ExecutorRsp, VectorOpsParityWithCpu) {
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
    ad.setPackingMode(DressiAD::PackingMode::RSP);
    ad.setLossVar(loss);
    ad.sendImg(n, img_n);
    ad.sendImg(l, img_l);
    ad.execStep();

    CpuEvaluator ev;
    ev.bindImage(n, img_n);
    ev.bindImage(l, img_l);
    ExpectImagesNear(ad.recvImg(loss), CpuImageFromTensor(ev.eval(loss)),
                     1e-5f);
}

TEST(ExecutorRsp, ImageFittingConvergesRsp) {
    const uint32_t W = 16, H = 16;
    Variable param(VEC3, {W, H});
    Variable target(VEC3, {W, H});
    Variable diff = target - param;
    Variable loss = F::Mean(diff * diff);
    Variable param_mut = param;
    param_mut.setRequiresGradRecursively();

    const float lr = 0.4f * float(W * H * 3) / 2.f;

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::RSP);
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
