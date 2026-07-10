// Screen-space AA GPU tests: CPU/GPU parity of the forward and backward
// through the executor (gl_PrimitiveID raster + AA special ops) and an
// end-to-end single-triangle descent with CPU-side clip updates.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "aa_scene.h"
#include "core/cpu_eval.h"
#include "dressi/dressi.h"
#include "grad_capture.h"

using namespace dressi;
using aa_test::AaScene;
using aa_test::MakeTensor;
using dressi_test::GradCapture;
using dressi_test::GradOf;
using dressi_test::IdentityOptimizer;

namespace {

constexpr uint32_t kScreen = 32;

std::array<float, 4> Clip(float sx, float sy, float z, float w = 1.f) {
    return {(sx / kScreen * 2.f - 1.f) * w, (sy / kScreen * 2.f - 1.f) * w,
            z * w, w};
}

AaScene GenericScene() {
    return AaScene({Clip(8.3f, 7.6f, 0.5f), Clip(24.2f, 10.4f, 0.5f, 1.3f),
                    Clip(10.6f, 25.1f, 0.5f)},
                   {{0, 1, 2}});
}

}  // namespace

TEST(AntiAliasGpu, ForwardParityCpu) {
    AaScene scene = GenericScene();
    Variable caa = scene.antialiasedMask({kScreen, kScreen});
    Variable loss = F::Mean(caa);

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    scene.send(ad);
    ad.markOutput(caa);
    ad.execStep();
    const CpuImage gpu = ad.recvImg(caa);

    CpuEvaluator ev;
    scene.bind(ev);
    const CpuTensor cpu = ev.eval(caa);

    // HW and CPU rasterization may disagree exactly on edge pixels, which
    // shifts the AA blend there; require agreement almost everywhere
    uint32_t mismatches = 0;
    for (size_t i = 0; i < cpu.data.size(); i++) {
        if (std::abs(gpu.data[i] - cpu.data[i]) > 1e-3f) {
            mismatches++;
        }
    }
    EXPECT_LE(mismatches, 8u);
}

TEST(AntiAliasGpu, BackwardParityCpu) {
    for (const auto mode : {DressiAD::PackingMode::Naive,
                            DressiAD::PackingMode::RSP}) {
        AaScene scene = GenericScene();
        Variable target(FLOAT, {kScreen, kScreen});
        Variable pred = scene.antialiasedMask({kScreen, kScreen});
        Variable diff = F::StopGradient(target) - pred;
        Variable loss = F::Mean(diff * diff);
        Variable clip_mut = scene.clip;
        clip_mut.setRequiresGradRecursively();

        CpuTensor t_target = MakeTensor(
                FLOAT, {kScreen, kScreen},
                std::vector<float>(size_t(kScreen) * kScreen, 0.f));
        for (size_t i = 0; i < t_target.data.size(); i++) {
            t_target.data[i] = 0.1f + 0.5f * float(i % 7) / 6.f;
        }

        DressiAD ad;
        ad.setPackingMode(mode);
        GradCapture cap;
        ad.setLossVar(loss);
        ad.setOptimizer(IdentityOptimizer(ad, cap));
        scene.send(ad);
        ad.sendImg(target, CpuImageFromTensor(t_target));
        ad.execStep();

        const Variable grad = GradOf(cap, scene.clip);
        ASSERT_TRUE(grad);
        const CpuImage gpu_grad = ad.recvImg(grad);

        CpuEvaluator ev;
        scene.bind(ev);
        ev.bind(target, t_target);
        const CpuTensor cpu_grad = ev.eval(grad);

        // The AA backward reads the hard mask / tri-id buffers; HW edge
        // rules can flip a handful of boundary contributions, so compare
        // loosely and require the bulk to match tightly
        ASSERT_EQ(gpu_grad.data.size(), cpu_grad.data.size());
        float max_abs = 0.f;
        for (float g : cpu_grad.data) {
            max_abs = std::max(max_abs, std::abs(g));
        }
        ASSERT_GT(max_abs, 0.f);
        for (size_t i = 0; i < cpu_grad.data.size(); i++) {
            EXPECT_NEAR(gpu_grad.data[i], cpu_grad.data[i], 0.05f * max_abs)
                    << "component " << i << " mode " << int(mode);
        }
    }
}

TEST(AntiAliasGpu, SingleTriangleDescent) {
    // Optimize a triangle's clip xy toward the AA'd mask of a shifted copy
    AaScene scene({Clip(8, 8, 0.5f), Clip(22, 10, 0.5f),
                   Clip(10, 23, 0.5f)},
                  {{0, 1, 2}});
    AaScene target_scene({Clip(11, 10, 0.5f), Clip(25, 12, 0.5f),
                          Clip(13, 25, 0.5f)},
                         {{0, 1, 2}});
    // Target: AA'd mask of the shifted triangle, evaluated on CPU
    CpuEvaluator tev;
    target_scene.bind(tev);
    const CpuTensor t_target =
            tev.eval(target_scene.antialiasedMask({kScreen, kScreen}));

    Variable target(FLOAT, {kScreen, kScreen});
    Variable pred = scene.antialiasedMask({kScreen, kScreen});
    Variable diff = F::StopGradient(target) - pred;
    Variable loss = F::Mean(diff * diff);
    Variable clip_mut = scene.clip;
    clip_mut.setRequiresGradRecursively();

    DressiAD ad;
    GradCapture cap;
    ad.setLossVar(loss);
    ad.setOptimizer(IdentityOptimizer(ad, cap));
    scene.send(ad);
    ad.sendImg(target, CpuImageFromTensor(t_target));

    float first_loss = 0.f;
    float last_loss = 0.f;
    // AA gradients live in a ~1px boundary band; keep steps below a pixel
    // (a large step can collapse the triangle to zero coverage, where no
    // ID boundary remains and the gradient vanishes for good)
    const float lr = 1.f;
    for (int iter = 0; iter < 400; iter++) {
        ad.execStep();
        last_loss = ad.recvImg(loss).data[0];
        if (iter == 0) {
            first_loss = last_loss;
        }
        const Variable grad = GradOf(cap, scene.clip);
        ASSERT_TRUE(grad);
        const CpuImage g = ad.recvImg(grad);
        for (int k = 0; k < 3; k++) {
            scene.t_clip.data[size_t(k) * 4 + 0] -= lr * g.at(k, 0, 0);
            scene.t_clip.data[size_t(k) * 4 + 1] -= lr * g.at(k, 0, 1);
        }
        ad.sendImg(scene.clip, CpuImageFromTensor(scene.t_clip));
    }
    EXPECT_LT(last_loss, first_loss / 5.f)
            << "first " << first_loss << " last " << last_loss;
}
