// Screen-space AA (Dr.Hair Eq.3-5) CPU tests: forward blending profile and
// finite-difference gradient checks w.r.t. the image and the clip
// positions (image/tri-id held fixed so central differences probe exactly
// the edge-distance path the analytic backward computes).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "aa_scene.h"
#include "core/build_backward.h"
#include "core/cpu_eval.h"
#include "core/cpu_raster.h"
#include "dressi/dressi.h"

using namespace dressi;
using aa_test::AaScene;
using aa_test::MakeTensor;

namespace {

constexpr uint32_t kScreen = 16;

std::array<float, 4> Clip(float sx, float sy, float z, float w = 1.f) {
    return {(sx / kScreen * 2.f - 1.f) * w, (sy / kScreen * 2.f - 1.f) * w,
            z * w, w};
}

struct Binding {
    Variable var;
    CpuTensor value;
};

float EvalLoss(const Variable& loss, const std::vector<Binding>& binds) {
    CpuEvaluator ev;
    for (const auto& b : binds) {
        ev.bind(b.var, b.value);
    }
    return ev.eval(loss).data[0];
}

void CheckGrad(const Variable& loss, std::vector<Binding> binds,
               const Variable& wrt, float h = 1e-3f, float rel_tol = 2e-2f,
               float abs_tol = 2e-3f, bool zero_z_grad = false) {
    Variable wrt_mut = wrt;
    wrt_mut.setRequiresGradRecursively();
    auto [input_vars, input_grad_vars] = BuildBackward(loss);
    Variable grad_var(nullptr);
    for (size_t i = 0; i < input_vars.size(); i++) {
        if (input_vars[i] == wrt) {
            grad_var = input_grad_vars[i];
        }
    }
    ASSERT_TRUE(grad_var);
    ASSERT_EQ(grad_var.getVType(), wrt.getVType());
    ASSERT_EQ(grad_var.getImgSize(), wrt.getImgSize());

    CpuEvaluator ev;
    for (const auto& b : binds) {
        ev.bind(b.var, b.value);
    }
    const CpuTensor analytic = ev.eval(grad_var);

    size_t wrt_idx = binds.size();
    for (size_t i = 0; i < binds.size(); i++) {
        if (binds[i].var == wrt) {
            wrt_idx = i;
        }
    }
    ASSERT_LT(wrt_idx, binds.size());

    for (size_t i = 0; i < binds[wrt_idx].value.data.size(); i++) {
        const float orig = binds[wrt_idx].value.data[i];
        binds[wrt_idx].value.data[i] = orig + h;
        const float f_plus = EvalLoss(loss, binds);
        binds[wrt_idx].value.data[i] = orig - h;
        const float f_minus = EvalLoss(loss, binds);
        binds[wrt_idx].value.data[i] = orig;

        const float numeric = (f_plus - f_minus) / (2.f * h);
        const float tol = abs_tol + rel_tol * std::abs(numeric);
        EXPECT_NEAR(analytic.data[i], numeric, tol) << "component " << i;
        if (zero_z_grad && i % 4 == 2) {
            EXPECT_FLOAT_EQ(analytic.data[i], 0.f);
        }
    }
}

// AntiAlias over FIXED img/tri leaves (only the clip leaf is live)
struct FixedBufferAa {
    Variable img{FLOAT, {kScreen, kScreen}};
    Variable tri{FLOAT, {kScreen, kScreen}};
    Variable clip;
    Variable faces_f;
    CpuTensor t_img, t_tri, t_clip, t_faces_f;

    FixedBufferAa(const std::vector<std::array<float, 4>>& verts,
                  const std::vector<std::array<uint32_t, 3>>& faces,
                  std::vector<float> attr = {})
        : clip(VEC4, {uint32_t(verts.size()), 1}),
          faces_f(VEC3, {uint32_t(faces.size()), 1}) {
        std::vector<float> vc;
        for (const auto& v : verts) {
            vc.insert(vc.end(), v.begin(), v.end());
        }
        t_clip = MakeTensor(VEC4, {uint32_t(verts.size()), 1},
                            std::move(vc));
        std::vector<float> fc;
        for (const auto& f : faces) {
            fc.insert(fc.end(), {float(f[0]), float(f[1]), float(f[2])});
        }
        t_faces_f = MakeTensor(VEC3, {uint32_t(faces.size()), 1},
                               std::move(fc));
        // Rasterize the attribute + tri id once with the CPU rasterizer
        if (attr.empty()) {
            attr.assign(verts.size(), 1.f);
        }
        CpuTensor attr_t;
        attr_t.vtype = FLOAT;
        attr_t.size = t_clip.size;
        attr_t.data = std::move(attr);
        CpuTensor faces_t = t_faces_f;
        faces_t.vtype = IVEC3;
        t_img = RasterizeHardCpu(t_clip, attr_t, faces_t,
                                 {kScreen, kScreen});
        t_tri = RasterizeFaceIdCpu(t_clip, faces_t, {kScreen, kScreen});
    }

    Variable antialiased() const {
        return F::AntiAlias(img, tri, clip, faces_f);
    }

    std::vector<Binding> bindings() const {
        return {{img, t_img}, {tri, t_tri}, {clip, t_clip},
                {faces_f, t_faces_f}};
    }
};

}  // namespace

TEST(AntiAlias, ForwardInteriorBoundaryBackground) {
    // Generic-position triangle rasterized in-graph (CPU kernels)
    AaScene scene({Clip(4.3f, 3.8f, 0.5f), Clip(12.4f, 5.2f, 0.5f),
                   Clip(5.1f, 12.6f, 0.5f)},
                  {{0, 1, 2}});
    CpuEvaluator ev;
    scene.bind(ev);
    Variable mask = F::Rasterize(scene.clip, scene.ones, scene.faces_i,
                                 {kScreen, kScreen});
    Variable caa = scene.antialiasedMask({kScreen, kScreen});
    const CpuTensor t_mask = ev.eval(mask);
    const CpuTensor out = ev.eval(caa);

    uint32_t n_interior = 0, n_boundary = 0, n_background = 0;
    for (uint32_t y = 1; y + 1 < kScreen; y++) {
        for (uint32_t x = 1; x + 1 < kScreen; x++) {
            bool uniform = true;
            const float m = t_mask.data[size_t(y) * kScreen + x];
            for (int dy = -1; dy <= 1 && uniform; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    const float mn = t_mask.data[size_t(int(y) + dy) *
                                                         kScreen +
                                                 uint32_t(int(x) + dx)];
                    if (mn != m) {
                        uniform = false;
                        break;
                    }
                }
            }
            const float v = out.data[size_t(y) * kScreen + x];
            if (uniform) {
                // All neighbors share the tri id: c_aa == c
                EXPECT_NEAR(v, m, 1e-6f) << "pixel " << x << "," << y;
                (m > 0.5f ? n_interior : n_background)++;
            } else {
                // Mixed neighborhood: value stays in [0,1]; pixels whose
                // every crossing pair clamps at r = 1 keep c exactly
                EXPECT_GE(v, 0.f);
                EXPECT_LE(v, 1.f);
                if (v > 1e-6f && v < 1.f - 1e-6f) {
                    n_boundary++;
                }
            }
        }
    }
    EXPECT_GT(n_interior, 0u);
    EXPECT_GT(n_boundary, 0u);
    EXPECT_GT(n_background, 0u);
}

TEST(AntiAlias, CheckGradImg) {
    FixedBufferAa fx({Clip(4.3f, 3.8f, 0.5f), Clip(12.4f, 5.2f, 0.5f),
                      Clip(5.1f, 12.6f, 0.5f)},
                     {{0, 1, 2}});
    Variable target(FLOAT, {kScreen, kScreen});
    CpuTensor t_target = MakeTensor(
            FLOAT, {kScreen, kScreen},
            std::vector<float>(size_t(kScreen) * kScreen, 0.f));
    for (size_t i = 0; i < t_target.data.size(); i++) {
        t_target.data[i] = 0.1f + 0.6f * float(i % 7) / 6.f;
    }
    Variable diff = F::StopGradient(target) - fx.antialiased();
    Variable loss = F::Mean(diff * diff);
    auto binds = fx.bindings();
    binds.push_back({target, t_target});
    CheckGrad(loss, binds, fx.img, 1e-3f, 1e-3f, 1e-5f);  // linear: exact
}

TEST(AntiAlias, CheckGradVtxClip) {
    // Generic position, one w != 1 vertex; img/tri fixed so FD probes only
    // the edge-distance path
    FixedBufferAa fx({Clip(4.3f, 3.8f, 0.5f), Clip(12.4f, 5.2f, 0.5f, 1.3f),
                      Clip(5.1f, 12.6f, 0.5f)},
                     {{0, 1, 2}});
    Variable target(FLOAT, {kScreen, kScreen});
    CpuTensor t_target = MakeTensor(
            FLOAT, {kScreen, kScreen},
            std::vector<float>(size_t(kScreen) * kScreen, 0.35f));
    Variable diff = F::StopGradient(target) - fx.antialiased();
    Variable loss = F::Mean(diff * diff);
    auto binds = fx.bindings();
    binds.push_back({target, t_target});
    CheckGrad(loss, binds, fx.clip, 1e-3f, 2e-2f, 2e-3f,
              /*zero_z_grad=*/true);
}

TEST(AntiAlias, OcclusionBoundaryFrontEdgeOwnsGradient) {
    // Back triangle covers the whole screen (its edges are off-screen);
    // front triangle sits in the middle with a different shade. Every ID
    // boundary is the front silhouette over the back surface, so the back
    // vertices must receive exactly zero gradient (owner-preference
    // fallback picks the front triangle's separating edge).
    FixedBufferAa fx({// back (face 0), z = 0.8, shade 0.2
                      Clip(-40, -40, 0.8f), Clip(100, -40, 0.8f),
                      Clip(-40, 100, 0.8f),
                      // front (face 1), z = 0.2, shade 0.9
                      Clip(5.2f, 4.7f, 0.2f), Clip(11.6f, 6.3f, 0.2f),
                      Clip(6.4f, 11.8f, 0.2f)},
                     {{0, 1, 2}, {3, 4, 5}},
                     {0.2f, 0.2f, 0.2f, 0.9f, 0.9f, 0.9f});
    Variable loss = F::Mean(fx.antialiased());
    Variable clip_mut = fx.clip;
    clip_mut.setRequiresGradRecursively();
    auto [input_vars, input_grad_vars] = BuildBackward(loss);
    ASSERT_EQ(input_vars.size(), 1u);

    CpuEvaluator ev;
    for (const auto& b : fx.bindings()) {
        ev.bind(b.var, b.value);
    }
    const CpuTensor g = ev.eval(input_grad_vars[0]);
    float back_mag = 0.f, front_mag = 0.f;
    for (uint32_t v = 0; v < 3; v++) {
        for (uint32_t c = 0; c < 4; c++) {
            back_mag += std::abs(g.data[size_t(v) * 4 + c]);
            front_mag += std::abs(g.data[size_t(v + 3) * 4 + c]);
        }
    }
    EXPECT_FLOAT_EQ(back_mag, 0.f);
    EXPECT_GT(front_mag, 1e-4f);
}
