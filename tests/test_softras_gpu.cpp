// HardSoftRas GPU tests: CPU/GPU forward parity, gradient parity through
// the executor (Naive and RSP) and an end-to-end single-triangle
// silhouette fit driven by the CPU-side update loop.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "core/cpu_eval.h"
#include "core/cpu_raster.h"
#include "dressi/dressi.h"
#include "softras_scene.h"

using namespace dressi;
using softras_test::ClipOfScreen;
using softras_test::MakeTensor;
using softras_test::SoftScene;

namespace {

constexpr uint32_t kScreen = 32;
constexpr float kRadius = 3.f;

std::array<float, 4> Clip(float sx, float sy, float z, float w = 1.f) {
    return ClipOfScreen(sx, sy, z, kScreen, w);
}

SoftScene GenericScene() {
    return SoftScene({Clip(8.3f, 7.6f, 0.5f), Clip(24.2f, 10.4f, 0.5f, 1.3f),
                      Clip(10.6f, 25.1f, 0.5f)},
                     {{0, 1, 2}}, 1.6f);
}

// Hard 0/1 coverage mask of a triangle via the CPU reference rasterizer
CpuTensor HardMask(const CpuTensor& clip, const CpuTensor& faces) {
    CpuTensor ones;
    ones.vtype = FLOAT;
    ones.size = clip.size;
    ones.data.assign(clip.size.w, 1.f);
    return RasterizeHardCpu(clip, ones, faces, {kScreen, kScreen});
}

// Captures the backward-graph gradient variables from the optimizer hook
struct GradCapture {
    Variables xs, gxs;
    bool marked = false;
};

DressiAD::Optimizer IdentityOptimizer(DressiAD& ad, GradCapture& cap) {
    return [&ad, &cap](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] + F::Float(0.f) * gxs[i]);
        }
        if (!cap.marked) {
            cap.xs = xs;
            cap.gxs = gxs;
            for (const auto& g : gxs) {
                ad.markOutput(g);
            }
            cap.marked = true;
        }
        return updated;
    };
}

Variable GradOf(const GradCapture& cap, const Variable& wrt) {
    for (size_t i = 0; i < cap.xs.size(); i++) {
        if (cap.xs[i] == wrt) {
            return cap.gxs[i];
        }
    }
    return Variable(nullptr);
}

}  // namespace

TEST(SoftRasGpu, ForwardParityCpu) {
    SoftScene scene = GenericScene();
    Variable out = scene.rasterize({kScreen, kScreen}, kRadius);
    Variable loss = F::Mean(F::GetX(out) * F::GetW(out));

    DressiAD ad;
    ad.setPackingMode(DressiAD::PackingMode::Naive);
    ad.setLossVar(loss);
    scene.send(ad);
    ad.execStep();
    const CpuImage gpu = ad.recvImg(out);

    CpuEvaluator ev;
    scene.bind(ev);
    const CpuTensor cpu = ev.eval(out);

    // HW and CPU fill rules may disagree exactly on the soft-coverage
    // boundary; compare only pixels whose coverage flags agree and count
    // the disagreements.
    uint32_t cover_mismatch = 0;
    for (uint32_t y = 0; y < kScreen; y++) {
        for (uint32_t x = 0; x < kScreen; x++) {
            const size_t o = (size_t(y) * kScreen + x) * 4;
            const bool ccov = cpu.data[o + 3] > 0.5f;
            const bool gcov = gpu.data[o + 3] > 0.5f;
            if (ccov != gcov) {
                cover_mismatch++;
                continue;
            }
            for (uint32_t c = 0; c < 4; c++) {
                ASSERT_NEAR(gpu.data[o + c], cpu.data[o + c], 1e-3f)
                        << "pixel " << x << "," << y << " ch " << c;
            }
        }
    }
    EXPECT_LE(cover_mismatch, 8u);
}

TEST(SoftRasGpu, BackwardParityCpu) {
    for (const auto mode : {DressiAD::PackingMode::Naive,
                            DressiAD::PackingMode::RSP}) {
        SoftScene scene = GenericScene();
        Variable target(FLOAT, {kScreen, kScreen});
        Variable pred = softras_test::SoftSilhouette(
                scene.rasterize({kScreen, kScreen}, kRadius), kRadius);
        Variable diff = F::StopGradient(target) - pred;
        Variable loss = F::Mean(diff * diff);
        Variable hard_mut = scene.hard_clip;
        hard_mut.setRequiresGradRecursively();

        CpuTensor t_target = MakeTensor(
                FLOAT, {kScreen, kScreen},
                std::vector<float>(size_t(kScreen) * kScreen, 0.f));
        for (size_t i = 0; i < t_target.data.size(); i++) {
            t_target.data[i] = 0.15f + 0.4f * float(i % 5) / 4.f;
        }

        DressiAD ad;
        ad.setPackingMode(mode);
        GradCapture cap;
        ad.setLossVar(loss);
        ad.setOptimizer(IdentityOptimizer(ad, cap));
        scene.send(ad);
        ad.sendImg(target, CpuImageFromTensor(t_target));
        ad.execStep();

        const Variable grad = GradOf(cap, scene.hard_clip);
        ASSERT_TRUE(grad);
        const CpuImage gpu_grad = ad.recvImg(grad);

        CpuEvaluator ev;
        scene.bind(ev);
        ev.bind(target, t_target);
        const CpuTensor cpu_grad = ev.eval(grad);

        ASSERT_EQ(gpu_grad.data.size(), cpu_grad.data.size());
        for (size_t i = 0; i < cpu_grad.data.size(); i++) {
            EXPECT_NEAR(gpu_grad.data[i], cpu_grad.data[i],
                        1e-4f + 2e-3f * std::abs(cpu_grad.data[i]))
                    << "component " << i << " mode " << int(mode);
        }
    }
}

TEST(SoftRasGpu, SingleTriangleSilhouetteFit) {
    // Optimize a triangle's clip xy toward a shifted target silhouette
    SoftScene scene({Clip(8, 8, 0.5f), Clip(22, 10, 0.5f),
                     Clip(10, 23, 0.5f)},
                    {{0, 1, 2}}, 1.6f);
    CpuTensor target_clip = scene.t_hard_clip;
    // Shift the target by (+4, +3) pixels in screen space (w = 1)
    for (int k = 0; k < 3; k++) {
        target_clip.data[size_t(k) * 4 + 0] += 4.f / kScreen * 2.f;
        target_clip.data[size_t(k) * 4 + 1] += 3.f / kScreen * 2.f;
    }
    const CpuTensor t_target = HardMask(target_clip, scene.t_faces_tex);

    Variable target(FLOAT, {kScreen, kScreen});
    Variable pred = softras_test::SoftSilhouette(
            scene.rasterize({kScreen, kScreen}, kRadius), kRadius);
    Variable diff = F::StopGradient(target) - pred;
    Variable loss = F::Mean(diff * diff);
    Variable hard_mut = scene.hard_clip;
    hard_mut.setRequiresGradRecursively();

    DressiAD ad;
    GradCapture cap;
    ad.setLossVar(loss);
    ad.setOptimizer(IdentityOptimizer(ad, cap));
    scene.send(ad);
    ad.sendImg(target, CpuImageFromTensor(t_target));

    float first_loss = 0.f;
    float last_loss = 0.f;
    const float lr = 0.5f;
    for (int iter = 0; iter < 200; iter++) {
        ad.execStep();
        last_loss = ad.recvImg(loss).data[0];
        if (iter == 0) {
            first_loss = last_loss;
        }
        const Variable grad = GradOf(cap, scene.hard_clip);
        ASSERT_TRUE(grad);
        const CpuImage g = ad.recvImg(grad);
        for (int k = 0; k < 3; k++) {
            // Update clip xy only (fixed depth and w for the test)
            scene.t_hard_clip.data[size_t(k) * 4 + 0] -= lr * g.at(k, 0, 0);
            scene.t_hard_clip.data[size_t(k) * 4 + 1] -= lr * g.at(k, 0, 1);
        }
        scene.rebuildSoft();
        ad.sendImg(scene.hard_clip,
                   CpuImageFromTensor(scene.t_hard_clip));
        ad.sendImg(scene.soft_clip,
                   CpuImageFromTensor(scene.t_soft_clip));
    }
    EXPECT_LT(last_loss, first_loss / 10.f)
            << "first " << first_loss << " last " << last_loss;
}
