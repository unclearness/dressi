// Sphere -> bunny in two stages, from a UV-unwrapped sphere.
//
//   Phase 1 (shape).  A position-WELDED copy of the lat-long UV sphere is
//   deformed toward the bunny silhouette from several views with the
//   screen-space AA technique (Dr.Hair, arXiv:2403.17496; the same path as
//   silhouette_optimization --technique=aa): hard rasterization plus a
//   differentiable blend across triangle-ID boundaries carries
//   vertex-position gradients, regularized by a uniform Laplacian and a
//   normal-consistency term and stepped by in-graph Adam. Welding is what
//   lets the sphere's seam/pole DUPLICATE vertices move as one -- optimizing
//   the unwelded UV mesh directly would tear the seam open.
//
//   Phase 2 (texture).  The optimized welded positions are scattered back
//   onto the UV-carrying (unwelded) sphere, freezing its shape. A texture
//   atlas in the sphere's own UV space is then optimized (deferred unlit,
//   Sobol-jittered like texture_optimization) so the textured sphere matches
//   in-graph renders of the textured bunny. The recovered atlas is the
//   bunny's appearance re-parameterized onto the fitted sphere.
//
// Every iteration lives on the GPU (the paper's transfer model). The result
// is a single textured mesh (out_dir/textured_sphere.obj + recovered_texture)
// reconstructed from silhouettes and images alone.

#include "run.h"

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "../common/asset_utils.h"
#include "../common/bench.h"
#include "../common/geometry_utils.h"
#include "../common/sobol.h"
#include "dressi/dressi.h"

using namespace dressi;
using namespace dressi_examples;

namespace {

struct Options {
    std::string data_dir = "data/bunny";
    std::string out_dir = "shapetex_out";
    uint32_t n_views = 8;
    int shape_iters = 200;
    int texture_iters = 300;
    uint32_t stacks = 32;
    uint32_t slices = 64;
    uint32_t ico_level = 4;  // icosphere subdivision for the shape phase
    uint32_t shape_screen = 128;
    uint32_t texture_screen = 256;
    uint32_t samples = 8;  // stochastic AA backward samples per face
    int view_interval = 1;
    bool no_view = false;
};

Options ParseArgs(const std::vector<std::string>& args) {
    Options o;
    for (const std::string& a : args) {
        std::string key, value;
        if (a.rfind("--", 0) == 0) {
            if (const size_t eq = a.find('='); eq != std::string::npos) {
                key = a.substr(0, eq);
                value = a.substr(eq + 1);
            } else {
                key = a;  // bare flag, e.g. --no-view
            }
        }
        if (key == "--data-dir") {
            o.data_dir = value;
        } else if (key == "--out-dir") {
            o.out_dir = value;
        } else if (key == "--views") {
            o.n_views = uint32_t(std::stoul(value));
        } else if (key == "--shape-iters") {
            o.shape_iters = std::stoi(value);
        } else if (key == "--texture-iters") {
            o.texture_iters = std::stoi(value);
        } else if (key == "--stacks") {
            o.stacks = uint32_t(std::stoul(value));
        } else if (key == "--slices") {
            o.slices = uint32_t(std::stoul(value));
        } else if (key == "--sphere-level") {
            o.ico_level = uint32_t(std::stoul(value));
        } else if (key == "--samples") {
            o.samples = uint32_t(std::stoul(value));
        } else if (key == "--view-interval") {
            o.view_interval = std::stoi(value);
        } else if (key == "--no-view") {
            o.no_view = value.empty() || value == "1" || value == "true";
        } else if (a.rfind("--", 0) != 0) {
            o.data_dir = a;  // positional data dir
        } else {
            throw DressiError("unknown option: " + a);
        }
    }
    return o;
}

float MaskIoU(const CpuImage& a, const CpuImage& b) {
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        const bool pa = a.data[i] > 0.5f;
        const bool pb = b.data[i] > 0.5f;
        inter += (pa && pb);
        uni += (pa || pb);
    }
    return uni ? float(inter) / float(uni) : 1.f;
}

std::vector<Mat4> BuildCameras(uint32_t n_views) {
    const Mat4 proj =
            PerspectiveVk(45.f * 3.14159265f / 180.f, 1.f, 0.1f, 10.f);
    std::vector<Mat4> mvps(n_views);
    for (uint32_t v = 0; v < n_views; v++) {
        const float ang = 2.f * 3.14159265f * float(v) / float(n_views);
        const float elev = (v % 2 == 0) ? 0.5f : -0.3f;
        const std::array<float, 3> eye = {1.6f * std::sin(ang), elev,
                                          1.6f * std::cos(ang)};
        mvps[v] = Mul(proj, LookAt(eye, {0, 0, 0}, {0, 1, 0}));
    }
    return mvps;
}

// Viewers are created once by the caller and shared across both phases: the
// Android host never unregisters a stream, so per-phase viewers would pile up
// stale stream buttons and freeze the surface on the phase switch. `tex` is
// only used by the texture phase (nullptr in phase 1).
struct Viewers {
    IViewer* target = nullptr;
    IViewer* pred = nullptr;
    IViewer* tex = nullptr;
    bool open() const {
        return target && target->valid() && pred && pred->valid();
    }
};

// -------------------------------------------------------------------------
// Phase 1: multi-view AA silhouette optimization of the icosphere toward the
// bunny. Returns the final icosphere vertex positions {V,1,3}.
// -------------------------------------------------------------------------
CpuImage OptimizeShape(const Mesh& bunny, const Mesh& sphere_w,
                       const std::vector<Mat4>& mvps, ImgSize screen,
                       const Options& opt, ExampleHost& host,
                       const Viewers& vw) {
    const uint32_t n_verts = sphere_w.numVertices();
    const uint32_t n_faces = sphere_w.numFaces();
    const uint32_t n_views = uint32_t(mvps.size());
    const float w_lap = 0.5f, w_nrm = 0.5f, lr = 0.01f;

    const CpuImage b_vf = VertexFacesTex(bunny.faces, bunny.numVertices());
    const CpuImage s_vf = VertexFacesTex(sphere_w.faces, n_verts);
    const CpuImage s_vn = VertexNeighborsTex(sphere_w.faces, n_verts);
    const CpuImage s_fa = FaceNeighborsTex(sphere_w.faces);

    Variable b_ones(FLOAT, {bunny.numVertices(), 1});
    Variable b_faces_i(IVEC3, {bunny.numFaces(), 1});
    Variable b_faces_f(VEC3, {bunny.numFaces(), 1});
    Variable b_vtx_faces(FLOAT, b_vf.getImgSize());
    Variable s_ones(FLOAT, {n_verts, 1});
    Variable s_faces_i(IVEC3, {n_faces, 1});
    Variable s_faces_f(VEC3, {n_faces, 1});
    Variable s_vtx_faces(FLOAT, s_vf.getImgSize());
    Variable s_vtx_nbrs(FLOAT, s_vn.getImgSize());
    Variable s_face_adj(VEC3, {n_faces, 1});

    Variable pos(VEC3, {n_verts, 1});
    Variable adam_m(VEC3, {n_verts, 1});
    Variable adam_v(VEC3, {n_verts, 1});
    Variable adam_t(FLOAT, {1, 1});
    Variable zero_seed(FLOAT, {1, 1});

    const auto build_pred = [&](const Variable& clip) -> Variable {
        Variable mask = F::Rasterize(clip, s_ones, s_faces_i, screen);
        Variable tri = F::RasterizeFaceId(clip, s_ones, s_faces_i, screen);
        return F::AntiAlias(mask, tri, clip, s_faces_f, s_vtx_faces, adam_t,
                            opt.samples);
    };
    const auto build_target = [&](const Variable& tclip) -> Variable {
        Variable m = F::Rasterize(tclip, b_ones, b_faces_i, screen);
        Variable t = F::RasterizeFaceId(tclip, b_ones, b_faces_i, screen);
        return F::StopGradient(F::AntiAlias(m, t, tclip, b_faces_f,
                                            b_vtx_faces, zero_seed, 0));
    };

    std::vector<Variable> tgt_clips;
    Variables targets, preds, view_losses;
    for (uint32_t v = 0; v < n_views; v++) {
        tgt_clips.emplace_back(VEC4, ImgSize{bunny.numVertices(), 1});
        targets.push_back(build_target(tgt_clips.back()));
        preds.push_back(build_pred(TransformToClipVar(pos, mvps[v])));
        Variable diff = targets[v] - preds[v];
        view_losses.push_back(diff * diff *
                              (1.f / float(screen.w * screen.h)));
    }
    Variable loss = F::SumPixelWise(view_losses);
    Variable pos_mut = pos;
    pos_mut.setRequiresGradRecursively();

    DressiAD ad;
    ad.setLossVar(loss);
    bool state_registered = false;
    ad.setOptimizer([&](Variables xs, Variables gxs) {
        DRESSI_CHECK(xs.size() == 1 && xs[0] == pos, "expected pos target");
        const float inv_cnt = 1.f / float(n_verts * 3);
        const auto rms = [&](const Variable& x) {
            return F::Sqrt(F::SumAll(x * x) * inv_cnt + 1e-24f);
        };
        Variable g = gxs[0];
        Variable data_rms = rms(g);
        Variable g_lap = pos - F::VertexNeighborMean(pos, s_vtx_nbrs);
        Variable g_nrm = F::NormalConsistencyVertexGrad(
                F::NormalConsistencyFaceTerm(pos, s_faces_f, s_face_adj),
                pos, s_faces_f, s_vtx_faces);
        g = g + g_lap * (data_rms / rms(g_lap) * w_lap) +
            g_nrm * (data_rms / rms(g_nrm) * w_nrm);
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        Variable t_new = adam_t + 1.f;
        Variable m_new = adam_m * b1 + g * (1.f - b1);
        Variable v_new = adam_v * b2 + g * g * (1.f - b2);
        Variable bc1 = 1.f - F::Pow(F::Float(b1), t_new);
        Variable bc2 = 1.f - F::Pow(F::Float(b2), t_new);
        Variable step =
                (m_new / bc1) / (F::Sqrt(v_new / bc2) + eps) * lr;
        if (!state_registered) {
            ad.addUpdate(adam_m, m_new);
            ad.addUpdate(adam_v, v_new);
            ad.addUpdate(adam_t, t_new);
            state_registered = true;
        }
        return Variables{pos - step};
    });
    for (uint32_t v = 0; v < n_views; v++) {
        ad.markOutput(targets[v]);
        ad.markOutput(preds[v]);
    }

    const CpuImage b_ones_img(bunny.numVertices(), 1, 1, 1.f);
    const CpuImage s_ones_img(n_verts, 1, 1, 1.f);
    ad.sendImg(b_ones, b_ones_img);
    ad.sendImg(b_faces_i, bunny.faces);
    ad.sendImg(b_faces_f, bunny.faces);
    ad.sendImg(b_vtx_faces, b_vf);
    ad.sendImg(s_ones, s_ones_img);
    ad.sendImg(s_faces_i, sphere_w.faces);
    ad.sendImg(s_faces_f, sphere_w.faces);
    ad.sendImg(s_vtx_faces, s_vf);
    ad.sendImg(s_vtx_nbrs, s_vn);
    ad.sendImg(s_face_adj, s_fa);
    ad.sendImg(pos, sphere_w.pos);
    ad.sendImg(adam_m, CpuImage(n_verts, 1, 3, 0.f));
    ad.sendImg(adam_v, CpuImage(n_verts, 1, 3, 0.f));
    ad.sendImg(adam_t, CpuImage(1, 1, 1, 0.f));
    ad.sendImg(zero_seed, CpuImage(1, 1, 1, 0.f));
    for (uint32_t v = 0; v < n_views; v++) {
        ad.sendImg(tgt_clips[v], TransformToClip(bunny.pos, mvps[v]));
    }

    const uint32_t tile_cols = 4;
    const uint32_t tile_rows = (n_views + tile_cols - 1) / tile_cols;
    (void)tile_rows;
    bool viewer_open = vw.open();
    CpuImage target_tile;

    using Clock = std::chrono::steady_clock;
    for (int iter = 0; iter < opt.shape_iters && !host.cancelled(); iter++) {
        ad.execStep();
        if (iter == 0) {
            std::vector<CpuImage> tgts;
            for (uint32_t v = 0; v < n_views; v++) {
                tgts.push_back(ad.recvImg(targets[v]));
            }
            target_tile = TileImages(tgts, tile_cols);
            SaveImagePng(opt.out_dir + "/phase1_targets.png", target_tile);
        }
        if (viewer_open && opt.view_interval > 0 &&
            iter % opt.view_interval == 0) {
            std::vector<CpuImage> prds;
            for (uint32_t v = 0; v < n_views; v++) {
                prds.push_back(ad.recvImg(preds[v]));
            }
            vw.pred->setTitle(
                    fmt::format("phase 1: silhouette  iter {}", iter));
            viewer_open = vw.pred->update(TileImages(prds, tile_cols)) &&
                          vw.target->update(target_tile);
        }
        if (iter % 50 == 0 || iter == opt.shape_iters - 1) {
            const CpuImage li = ad.recvImg(loss);
            double s = 0.0;
            for (float x : li.data) {
                s += x;
            }
            spdlog::info("[shape] iter {:4d}  loss {:.8f}", iter, s);
        }
    }

    // One extra step so pred/target readbacks are consistent, then report IoU
    ad.execStep();
    float mean_iou = 0.f;
    std::vector<CpuImage> final_preds;
    for (uint32_t v = 0; v < n_views; v++) {
        final_preds.push_back(ad.recvImg(preds[v]));
        mean_iou += MaskIoU(final_preds.back(), ad.recvImg(targets[v]));
    }
    mean_iou /= float(n_views);
    SaveImagePng(opt.out_dir + "/phase1_pred_last.png",
                 TileImages(final_preds, tile_cols));
    spdlog::info("[shape] done: mean silhouette IoU {:.4f}", mean_iou);
    return ad.recvImg(pos);
}

// -------------------------------------------------------------------------
// Phase 2: optimize a texture atlas on the (now fixed) UV sphere so its
// render matches the textured bunny. Mirrors texture_optimization but with
// two distinct meshes: bunny = target geometry+atlas, sphere = optimized.
// -------------------------------------------------------------------------
void OptimizeTexture(const Mesh& bunny, const CpuImage& bunny_atlas,
                     const Mesh& sphere, const std::vector<Mat4>& mvps,
                     ImgSize screen, ImgSize tex_size, const Options& opt,
                     ExampleHost& host, const Viewers& viewers) {
    const uint32_t n_views = uint32_t(mvps.size());
    const ImgSize bunny_tex = {bunny_atlas.width, bunny_atlas.height};

    // Per-mesh static leaves
    Variable s_uv(VEC2, {sphere.numVertices(), 1});
    Variable s_faces(IVEC3, {sphere.numFaces(), 1});
    Variable s_ones(FLOAT, {sphere.numVertices(), 1});
    Variable s_uv_clip(VEC4, {sphere.numVertices(), 1});
    Variable b_uv(VEC2, {bunny.numVertices(), 1});
    Variable b_faces(IVEC3, {bunny.numFaces(), 1});
    Variable b_ones(FLOAT, {bunny.numVertices(), 1});
    Variable b_uv_clip(VEC4, {bunny.numVertices(), 1});
    Variable gt_tex(VEC3, bunny_tex);  // bunny atlas (no grad)
    Variable tex(VEC3, tex_size);      // optimized sphere atlas

    const CpuImage s_uv_clip_img = UvAsClip(sphere.uv);
    const CpuImage b_uv_clip_img = UvAsClip(bunny.uv);
    const CpuImage s_ones_img(sphere.numVertices(), 1, 1, 1.f);
    const CpuImage b_ones_img(bunny.numVertices(), 1, 1, 1.f);

    struct View {
        Variable s_clip{nullptr}, s_inv{nullptr};
        Variable b_clip{nullptr}, b_inv{nullptr};
        Variable target{nullptr}, pred{nullptr};
        CpuImage s_clip_img, b_clip_img;
    };
    std::vector<View> views(n_views);
    for (uint32_t v = 0; v < n_views; v++) {
        View& vw = views[v];
        vw.s_clip = Variable(VEC4, {sphere.numVertices(), 1});
        vw.s_inv = Variable(VEC4, {sphere.numVertices(), 1});
        vw.b_clip = Variable(VEC4, {bunny.numVertices(), 1});
        vw.b_inv = Variable(VEC4, {bunny.numVertices(), 1});
        vw.s_clip_img = TransformToClip(sphere.pos, mvps[v]);
        vw.b_clip_img = TransformToClip(bunny.pos, mvps[v]);
    }

    Variables view_losses;
    for (View& vw : views) {
        // Sphere (prediction) branch: G-buffer + differentiable texture
        Variable s_g_uv = F::Rasterize(vw.s_clip, s_uv, s_faces, screen);
        Variable s_mask = F::Rasterize(vw.s_clip, s_ones, s_faces, screen);
        Variable s_inv_uv =
                F::Rasterize(s_uv_clip, vw.s_inv, s_faces, tex_size);
        vw.pred = s_mask * F::Texture(tex, s_g_uv, s_inv_uv);
        // Bunny (target) branch: same deferred shading with the GT atlas
        Variable b_g_uv = F::Rasterize(vw.b_clip, b_uv, b_faces, screen);
        Variable b_mask = F::Rasterize(vw.b_clip, b_ones, b_faces, screen);
        Variable b_inv_uv =
                F::Rasterize(b_uv_clip, vw.b_inv, b_faces, bunny_tex);
        vw.target = b_mask * F::Texture(gt_tex, b_g_uv, b_inv_uv);
        // Supervise only where the bunny is visible (a sphere fringe that
        // the bunny does not cover would otherwise learn black)
        Variable diff = F::StopGradient(vw.target) - vw.pred;
        view_losses.push_back(F::Mean(F::StopGradient(b_mask) * diff * diff));
    }
    Variable loss = F::SumPixelWise(view_losses);
    Variable tex_mut = tex;
    tex_mut.setRequiresGradRecursively();

    const float lr =
            0.6f * float(screen.w * screen.h * 3) / (2.f * float(n_views));
    DressiAD ad;
    ad.setLossVar(loss);
    ad.setOptimizer([lr](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] - gxs[i] * lr);
        }
        return updated;
    });
    for (View& vw : views) {
        ad.markOutput(vw.target);
        ad.markOutput(vw.pred);
    }

    ad.sendImg(s_uv, sphere.uv);
    ad.sendImg(s_faces, sphere.faces);
    ad.sendImg(s_ones, s_ones_img);
    ad.sendImg(s_uv_clip, s_uv_clip_img);
    ad.sendImg(b_uv, bunny.uv);
    ad.sendImg(b_faces, bunny.faces);
    ad.sendImg(b_ones, b_ones_img);
    ad.sendImg(b_uv_clip, b_uv_clip_img);
    ad.sendImg(gt_tex, bunny_atlas);
    ad.sendImg(tex, CpuImage(tex_size.w, tex_size.h, 3, 0.f));  // black
    for (View& vw : views) {
        ad.sendImg(vw.s_clip, vw.s_clip_img);
        ad.sendImg(vw.s_inv, ScreenAttr(vw.s_clip_img, screen));
        ad.sendImg(vw.b_clip, vw.b_clip_img);
        ad.sendImg(vw.b_inv, ScreenAttr(vw.b_clip_img, screen));
    }

    const uint32_t tile_cols = 4;
    bool viewer_open = viewers.open();
    CpuImage target_tile;

    // Sub-pixel Sobol camera jitter (TAA style): covers texels densely over
    // time. Both meshes jitter by the SAME offset so target/pred stay aligned.
    const auto send_jitter = [&](int iter) {
        const auto s = Sobol2D(uint32_t(iter));
        const float jx = (s[0] - 0.5f) * 2.f / float(screen.w);
        const float jy = (s[1] - 0.5f) * 2.f / float(screen.h);
        const auto jitter = [&](const CpuImage& base) {
            CpuImage out = base;
            for (uint32_t i = 0; i < out.width; i++) {
                const float w = out.at(i, 0, 3);
                out.at(i, 0, 0) += jx * w;
                out.at(i, 0, 1) += jy * w;
            }
            return out;
        };
        for (View& vw : views) {
            ad.sendImg(vw.s_clip, jitter(vw.s_clip_img));
            ad.sendImg(vw.b_clip, jitter(vw.b_clip_img));
        }
    };

    for (int iter = 0; iter < opt.texture_iters && !host.cancelled();
         iter++) {
        send_jitter(iter);
        ad.execStep();
        if (iter == 0) {
            std::vector<CpuImage> tgts;
            for (uint32_t v = 0; v < n_views; v++) {
                tgts.push_back(ad.recvImg(views[v].target));
            }
            target_tile = TileImages(tgts, tile_cols);
            SaveImagePng(opt.out_dir + "/phase2_targets.png", target_tile);
        }
        if (viewer_open && opt.view_interval > 0 &&
            iter % opt.view_interval == 0) {
            std::vector<CpuImage> prds;
            for (uint32_t v = 0; v < n_views; v++) {
                prds.push_back(ad.recvImg(views[v].pred));
            }
            viewers.pred->setTitle(
                    fmt::format("phase 2: textured sphere  iter {}", iter));
            viewer_open =
                    viewers.pred->update(TileImages(prds, tile_cols)) &&
                    viewers.target->update(target_tile) &&
                    (!viewers.tex || viewers.tex->update(ad.recvImg(tex)));
        }
        if (iter % 100 == 0 || iter == opt.texture_iters - 1) {
            spdlog::info("[texture] iter {:4d}  loss {:.8f}", iter,
                         ad.recvImg(loss).data[0]);
        }
    }

    // Final renders at zero jitter (clean pixel centers)
    for (View& vw : views) {
        ad.sendImg(vw.s_clip, vw.s_clip_img);
        ad.sendImg(vw.b_clip, vw.b_clip_img);
    }
    ad.execStep();
    const CpuImage recovered = ad.recvImg(tex);
    SaveImagePng(opt.out_dir + "/recovered_texture.png", recovered);
    std::vector<CpuImage> final_preds, final_tgts;
    for (uint32_t v = 0; v < n_views; v++) {
        final_preds.push_back(ad.recvImg(views[v].pred));
        final_tgts.push_back(ad.recvImg(views[v].target));
    }
    SaveImagePng(opt.out_dir + "/phase2_pred_last.png",
                 TileImages(final_preds, tile_cols));

    // Write the fitted, textured mesh (references recovered_texture.png)
    SaveTexturedObjMesh(opt.out_dir + "/textured_sphere.obj", sphere,
                        "recovered_texture.png");
    spdlog::info("[texture] done: recovered atlas {}x{}", tex_size.w,
                 tex_size.h);
}

}  // namespace

int dressi_examples::RunShapeTextureOptimization(
        const std::vector<std::string>& args, ExampleHost& host) try {
    spdlog::cfg::load_env_levels();
    const Options opt = ParseArgs(args);
    std::filesystem::create_directories(opt.out_dir);

    // ------------------------- Assets / geometry -------------------------
    const Mesh bunny = LoadObjMesh(opt.data_dir + "/bunny.obj");
    const CpuImage bunny_atlas =
            LoadImageRgb(opt.data_dir + "/bunny-atlas.jpg");

    // Shape is optimized on a uniform ICOSPHERE (a well-conditioned
    // deformation cage -- a lat-long UV sphere's stretched pole triangles
    // deform poorly). The UV-unwrapped sphere only carries the texture
    // parameterization; it inherits the fitted shape by directional transfer.
    Mesh ico = GenerateIcosphere(opt.ico_level);
    for (float& p : ico.pos.data) {
        p *= 0.55f;  // initial shrink, matching silhouette_optimization
    }
    Mesh sphere = GenerateUvSphere(opt.stacks, opt.slices);  // carries UVs

    spdlog::info("bunny {}v/{}f, atlas {}x{}; shape icosphere {}v/{}f, "
                 "UV sphere {}v/{}f, {} views",
                 bunny.numVertices(), bunny.numFaces(), bunny_atlas.width,
                 bunny_atlas.height, ico.numVertices(), ico.numFaces(),
                 sphere.numVertices(), sphere.numFaces(), opt.n_views);

    const std::vector<Mat4> mvps = BuildCameras(opt.n_views);

    // Viewers are created ONCE and shared by both phases (the Android host
    // never unregisters a stream, so per-phase viewers would accumulate stale
    // buttons and freeze the surface at the phase switch). All tiled outputs
    // are 2:1, so a single window aspect fits phase 1 and phase 2; the desktop
    // viewer scales any image into its window.
    std::unique_ptr<IViewer> v_target, v_pred, v_tex;
    if (!opt.no_view) {
        v_target = host.makeViewer(512, 256, "target (bunny)");
        v_pred = host.makeViewer(512, 256, "prediction (sphere)");
        v_tex = host.makeViewer(512, 256, "recovered atlas");
        v_target->setPosition(80, 80);
        v_pred->setPosition(80 + 512 + 16, 80);
        v_tex->setPosition(80, 80 + 256 + 46);
    }
    Viewers viewers{v_target.get(), v_pred.get(), v_tex.get()};

    // --------------------------- Phase 1: shape --------------------------
    const CpuImage ico_final = OptimizeShape(
            bunny, ico, mvps, {opt.shape_screen, opt.shape_screen}, opt,
            host, viewers);
    // Transfer the icosphere deformation onto the UV sphere by direction.
    // The map is a function of direction alone, so the UV sphere's seam/pole
    // duplicates (same direction) receive identical positions -> no seam tear.
    sphere.pos = TransferDeformationByDirection(ico.pos, ico.faces, ico_final,
                                                sphere.pos);
    SaveObjMesh(opt.out_dir + "/fitted_shape.obj", sphere);

    if (host.cancelled()) {
        spdlog::warn("cancelled after phase 1");
        return 0;
    }

    // -------------------------- Phase 2: texture -------------------------
    // A lat-long UV sphere unwraps to a 2:1 rectangle; size the atlas to match
    const ImgSize tex_size = {512, 256};
    OptimizeTexture(bunny, bunny_atlas, sphere, mvps,
                    {opt.texture_screen, opt.texture_screen}, tex_size, opt,
                    host, viewers);

    spdlog::info("outputs in {}/ (fitted_shape.obj, textured_sphere.obj, "
                 "recovered_texture.png, phase{{1,2}}_*.png)",
                 opt.out_dir);
    return 0;
} catch (const std::exception& e) {
    spdlog::error("error: {}", e.what());
    return 2;
}
