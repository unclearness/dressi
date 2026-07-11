// Multi-view environment-map optimization on DamagedHelmet: recover the
// (small) equirectangular HDR env map from rendered images while geometry,
// materials and the light stay at ground truth. The forward shader is
// exactly the viewer's; per view the G-buffer is rasterized once and shaded
// twice (GT env maps -> target, optimized env -> pred).
//
// Gradient paths into the env leaf (the paper has no recipe here; this is
// the documented Dressi-style solution):
//   - background pixels: EquirectSample(env, view_ray) -> exact bilinear
//     transpose (each env texel scans the screen; O(env * pixels), which is
//     why the OPTIMIZED env is a small leaf, 64x32 by default)
//   - diffuse IBL: EquirectSample(irr, n) -> transpose -> IrradianceConv
//     transpose (exact: the conv is a deterministic linear map) -> AvgPool
//     backward -> env
//   - specular IBL (PrefilterEnv) recomputes its FORWARD from the current
//     env every iteration but contributes no gradient (importance-sampled
//     prefilter has no transpose yet — documented deviation)
// Because the env leaf is dirty every iteration, the IBL precompute of the
// optimized branch is NOT pruned (it re-runs per iter); the GT branch stays
// static and is pruned as usual.

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "../common/pbr_graph.h"
#include "../common/vk_viewer.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

CpuImage Vec3Img(float x, float y, float z) {
    CpuImage img(1, 1, 3);
    img.data = {x, y, z};
    return img;
}

CpuImage DownsampleTo(CpuImage img, uint32_t max_w) {
    while (img.width > max_w) {
        img = DownsampleHalf(img);
    }
    return img;
}

// Reinhard + gamma for saving HDR maps as viewable PNGs
CpuImage TonemapForView(CpuImage img) {
    for (float& v : img.data) {
        v = std::pow(v / (v + 1.f), 1.f / 2.2f);
    }
    return img;
}

// Nearest upscale so the tiny env maps are inspectable
CpuImage Upscale(const CpuImage& img, uint32_t k) {
    CpuImage out(img.width * k, img.height * k, img.channels);
    for (uint32_t y = 0; y < out.height; y++) {
        for (uint32_t x = 0; x < out.width; x++) {
            for (uint32_t c = 0; c < img.channels; c++) {
                out.at(x, y, c) = img.at(x / k, y / k, c);
            }
        }
    }
    return out;
}

struct View {
    Variable clip{nullptr};
    PbrFrameLeaves frame;
    Variable target{nullptr};
    Variable pred{nullptr};
    CpuImage clip_img;
};

}  // namespace

int main(int argc, char* argv[]) try {
    spdlog::cfg::load_env_levels();

    std::string mesh_path = "data/DamagedHelmet/glTF/DamagedHelmet.gltf";
    std::string env_path = "data/suburban_garden_4k.exr";
    uint32_t size = 192, env_w = 64, n_views = 4;
    int n_iters = 1500;
    float lr = 0.05f;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg.rfind("--mesh=", 0) == 0) {
            mesh_path = arg.substr(7);
        } else if (arg.rfind("--env=", 0) == 0) {
            env_path = arg.substr(6);
        } else if (arg.rfind("--size=", 0) == 0) {
            size = uint32_t(std::stoi(arg.substr(7)));
        } else if (arg.rfind("--envres=", 0) == 0) {
            env_w = uint32_t(std::stoi(arg.substr(9)));
        } else if (arg.rfind("--views=", 0) == 0) {
            n_views = uint32_t(std::stoi(arg.substr(8)));
        } else if (arg.rfind("--iters=", 0) == 0) {
            n_iters = std::stoi(arg.substr(8));
        } else if (arg.rfind("--lr=", 0) == 0) {
            lr = std::stof(arg.substr(5));
        }
    }
    const ImgSize screen = {size, size};
    const std::string out_dir = "pbrenv_out";
    std::filesystem::create_directories(out_dir);

    // ------------------------- Load assets (CPU) -------------------------
    const GltfScene gltf = LoadGltfScene(mesh_path);
    const CpuImage env_gt_img = DownsampleTo(LoadImageExr(env_path), env_w);
    const ImgSize env_size = env_gt_img.getImgSize();
    const uint32_t tex_max = 512;
    const CpuImage m_albedo = DownsampleTo(SrgbToLinear(gltf.base_color),
                                           tex_max);
    const CpuImage m_mr = DownsampleTo(gltf.metallic_roughness, tex_max);
    const CpuImage m_nrm = DownsampleTo(gltf.normal_map, tex_max);
    const CpuImage m_ao = DownsampleTo(gltf.occlusion, tex_max);
    const CpuImage m_emis = DownsampleTo(SrgbToLinear(gltf.emissive),
                                         tex_max);
    spdlog::info("mesh: {} verts, {} faces; optimizing env at {}x{}",
                 gltf.mesh.numVertices(), gltf.mesh.numFaces(), env_size.w,
                 env_size.h);

    const uint32_t n_verts = gltf.mesh.numVertices();
    CpuImage uvh_img(n_verts, 1, 4);
    CpuImage tangent_img(n_verts, 1, 3);
    for (uint32_t v = 0; v < n_verts; v++) {
        uvh_img.at(v, 0, 0) = gltf.mesh.uv.at(v, 0, 0);
        uvh_img.at(v, 0, 1) = gltf.mesh.uv.at(v, 0, 1);
        uvh_img.at(v, 0, 2) = 1.f;
        uvh_img.at(v, 0, 3) = gltf.tangent.at(v, 0, 3);
        for (uint32_t c = 0; c < 3; c++) {
            tangent_img.at(v, 0, c) = gltf.tangent.at(v, 0, c);
        }
    }

    // ------------------------ Graph leaf variables ------------------------
    PbrGeometryLeaves geom;
    geom.uvh = Variable(VEC4, {n_verts, 1});
    geom.normal = Variable(VEC3, {n_verts, 1});
    geom.tangent = Variable(VEC3, {n_verts, 1});
    geom.wpos = Variable(VEC3, {n_verts, 1});
    geom.faces = Variable(IVEC3, {gltf.mesh.numFaces(), 1});

    PbrMaterialLeaves mat;
    mat.albedo = Variable(VEC3, m_albedo.getImgSize());
    mat.mr = Variable(VEC3, m_mr.getImgSize());
    mat.normal_map = Variable(VEC3, m_nrm.getImgSize());
    mat.ao = Variable(VEC3, m_ao.getImgSize());
    mat.emissive = Variable(VEC3, m_emis.getImgSize());
    // No texture gradients here -> never-uploaded dummy inverse-UV table
    Variable dummy_inv_uv(VEC4, m_albedo.getImgSize());

    Variable env_gt(VEC3, env_size);
    Variable env_opt(VEC3, env_size);
    const PbrIblMaps ibl_gt = BuildPbrIblMaps(env_gt);
    const PbrIblMaps ibl_opt = BuildPbrIblMaps(env_opt);

    const float fov = 45.f * 3.14159265f / 180.f;
    const float tan_half = std::tan(fov * 0.5f);
    const Mat4 proj = PerspectiveVk(fov, 1.f, 0.1f, 10.f);
    Variable light_pos(VEC3, {1, 1});
    Variable light_col(VEC3, {1, 1});

    // Alternate the elevation sign so the background rays cover both the
    // upper and lower halves of the sphere (all-above cameras never see
    // the sky and the recovery stays dark there)
    const auto view_eye = [&](uint32_t vi) {
        const float ang = 2.f * 3.14159265f * float(vi) / float(n_views);
        const float ey = (vi % 2 == 0) ? 0.85f : -0.85f;
        return std::array<float, 3>{2.2f * std::sin(ang) * 0.92f, ey,
                                    2.2f * std::cos(ang) * 0.92f};
    };

    std::vector<View> views(n_views);
    Variables view_losses;
    for (uint32_t vi = 0; vi < n_views; vi++) {
        View& view = views[vi];
        const std::array<float, 3> eye = view_eye(vi);
        const Mat4 vm = LookAt(eye, {0, 0, 0}, {0, 1, 0});
        view.clip = Variable(VEC4, {n_verts, 1});
        view.clip_img = TransformToClip(gltf.mesh.pos, Mul(proj, vm));

        view.frame.cam_pos = Variable(VEC3, {1, 1});
        view.frame.cam_fwd = Variable(VEC3, {1, 1});
        view.frame.cam_right = Variable(VEC3, {1, 1});
        view.frame.cam_up = Variable(VEC3, {1, 1});
        view.frame.light_pos = light_pos;
        view.frame.light_col = light_col;

        PbrGeometryLeaves vgeom = geom;
        vgeom.clip = view.clip;
        const PbrGBuffer gbuf = BuildPbrGBuffer(vgeom, screen);
        const PbrOutputs target = BuildPbrShade(gbuf, mat, view.frame,
                                                ibl_gt, dummy_inv_uv, screen);
        const PbrOutputs pred = BuildPbrShade(gbuf, mat, view.frame, ibl_opt,
                                              dummy_inv_uv, screen);
        view.target = target.ldr;
        view.pred = pred.ldr;
        Variable diff = F::StopGradient(target.mapped) - pred.mapped;
        view_losses.push_back(F::Mean(diff * diff));
    }
    Variable loss = F::SumPixelWise(view_losses);

    Variable env_mut = env_opt;
    env_mut.setRequiresGradRecursively();

    // ------------------- In-graph Adam (GPU-resident state) ---------------
    Variable adam_m(VEC3, env_size);
    Variable adam_v(VEC3, env_size);
    Variable adam_t(FLOAT, {1, 1});
    DressiAD ad;
    ad.setLossVar(loss);
    bool state_registered = false;
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        DRESSI_CHECK(xs.size() == 1 && xs[0] == env_opt,
                     "expected the env map as the only optimizer target");
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        Variable g = gxs[0];
        Variable t_new = adam_t + 1.f;
        Variable m_new = adam_m * b1 + g * (1.f - b1);
        Variable v_new = adam_v * b2 + g * g * (1.f - b2);
        Variable bc1 = 1.f - F::Pow(F::Float(b1), t_new);
        Variable bc2 = 1.f - F::Pow(F::Float(b2), t_new);
        Variable step = (m_new / bc1) / (F::Sqrt(v_new / bc2) + eps) * lr;
        if (!state_registered) {
            ad.addUpdate(adam_m, m_new);
            ad.addUpdate(adam_v, v_new);
            ad.addUpdate(adam_t, t_new);
            state_registered = true;
        }
        // Radiance stays non-negative (no upper bound: HDR)
        return Variables{F::Max(env_opt - step, F::Float(0.f))};
    });
    for (auto& view : views) {
        ad.markOutput(view.target);
        ad.markOutput(view.pred);
    }

    // ------------------------ One-time uploads ---------------------------
    ad.sendImg(geom.uvh, uvh_img);
    ad.sendImg(geom.normal, gltf.normal);
    ad.sendImg(geom.tangent, tangent_img);
    ad.sendImg(geom.wpos, gltf.mesh.pos);
    ad.sendImg(geom.faces, gltf.mesh.faces);
    ad.sendImg(mat.albedo, m_albedo);
    ad.sendImg(mat.mr, m_mr);
    ad.sendImg(mat.normal_map, m_nrm);
    ad.sendImg(mat.ao, m_ao);
    ad.sendImg(mat.emissive, m_emis);
    ad.sendImg(env_gt, env_gt_img);
    ad.sendImg(light_pos, Vec3Img(2.f, 2.f, 2.f));
    ad.sendImg(light_col, Vec3Img(20.f, 20.f, 20.f));
    // Gray HDR init around the scene's average brightness
    ad.sendImg(env_opt, CpuImage(env_size.w, env_size.h, 3, 0.3f));
    ad.sendImg(adam_m, CpuImage(env_size.w, env_size.h, 3, 0.f));
    ad.sendImg(adam_v, CpuImage(env_size.w, env_size.h, 3, 0.f));
    ad.sendImg(adam_t, CpuImage(1, 1, 1, 0.f));
    for (uint32_t vi = 0; vi < n_views; vi++) {
        View& view = views[vi];
        const std::array<float, 3> eye = view_eye(vi);
        const Mat4 vm = LookAt(eye, {0, 0, 0}, {0, 1, 0});
        ad.sendImg(view.clip, view.clip_img);
        ad.sendImg(view.frame.cam_pos, Vec3Img(eye[0], eye[1], eye[2]));
        ad.sendImg(view.frame.cam_fwd, Vec3Img(-vm[2], -vm[6], -vm[10]));
        ad.sendImg(view.frame.cam_right,
                   Vec3Img(vm[0] * tan_half, vm[4] * tan_half,
                           vm[8] * tan_half));
        ad.sendImg(view.frame.cam_up,
                   Vec3Img(vm[1] * tan_half, vm[5] * tan_half,
                           vm[9] * tan_half));
    }

    // --------------------------- Optimization ----------------------------
    // The env leaf is the only dirty leaf: re-send it? No — the optimizer
    // writes it in place, which marks it dynamic. Nothing else changes per
    // iteration, so no per-iteration uploads at all.
    // PSNR on Reinhard-tonemapped radiance (the raw HDR MSE is dominated
    // by the sun texels), over components the gradients actually touched
    const auto psnr_vs_gt = [&]() {
        const CpuImage rec = ad.recvImg(env_opt);
        double se = 0.0;
        size_t cnt = 0;
        for (size_t i = 0; i < rec.data.size(); i++) {
            if (std::abs(rec.data[i] - 0.3f) > 1e-3f) {  // touched by grads
                const double a = rec.data[i] / (1.f + rec.data[i]);
                const double b = env_gt_img.data[i] /
                                 (1.f + env_gt_img.data[i]);
                se += (a - b) * (a - b);
                cnt++;
            }
        }
        const double mse = cnt ? se / double(cnt) : 1.0;
        return std::pair<double, size_t>(
                10.0 * std::log10(1.0 / (mse + 1e-12)), cnt);
    };

    VkViewer viewer(screen.w * 2, screen.h, "pbr envmap optimization");
    bool viewer_open = viewer.valid();

    using Clock = std::chrono::steady_clock;
    std::vector<double> iter_ms;  // steady-state samples (>= warmup)
    const int warmup = 20;
    const auto median_ms = [&]() {
        if (iter_ms.empty()) {
            return 0.0;
        }
        std::vector<double> s = iter_ms;
        std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
        return s[s.size() / 2];
    };
    for (int iter = 0; iter < n_iters; iter++) {
        const auto t0 = Clock::now();
        ad.execStep();
        const double ms = std::chrono::duration<double, std::milli>(
                                  Clock::now() - t0)
                                  .count();
        if (iter >= warmup) {
            iter_ms.push_back(ms);
        }

        if (iter == 0) {
            SaveImagePng(out_dir + "/target_view0.png",
                         ad.recvImg(views[0].target));
            SaveImagePng(out_dir + "/render_first_view0.png",
                         ad.recvImg(views[0].pred));
        }
        if (viewer_open && iter % 20 == 0) {
            viewer.setTitle(fmt::format("pbr envmap opt  iter {}", iter));
            viewer_open = viewer.update(TileImages(
                    {ad.recvImg(views[0].target), ad.recvImg(views[0].pred)},
                    2));
        }
        if (iter % 200 == 0 || iter == n_iters - 1) {
            const auto [psnr, cnt] = psnr_vs_gt();
            spdlog::info(
                    "iter {:4d}  loss {:.6f}  env PSNR {:.2f} dB over {} "
                    "components  median {:.2f} ms/iter (excl. {}-iter "
                    "warmup)",
                    iter, ad.recvImg(loss).data[0], psnr, cnt, median_ms(),
                    warmup);
        }
    }

    // ------------------------------ Outputs -------------------------------
    const auto [psnr, cnt] = psnr_vs_gt();
    SaveImagePng(out_dir + "/env_gt.png",
                 Upscale(TonemapForView(env_gt_img), 8));
    SaveImagePng(out_dir + "/env_recovered.png",
                 Upscale(TonemapForView(ad.recvImg(env_opt)), 8));
    SaveImagePng(out_dir + "/render_last_view0.png",
                 ad.recvImg(views[0].pred));
    spdlog::info("final loss {:.6f}, env PSNR {:.2f} dB over {} components",
                 ad.recvImg(loss).data[0], psnr, cnt);
    spdlog::info("outputs in {}/", out_dir);
    return (psnr > 10.0 && cnt > 0) ? 0 : 1;
} catch (const std::exception& e) {
    std::printf("error: %s\n", e.what());
    return 2;
}
