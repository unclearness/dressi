#include "core/cpu_raster.h"

#include <algorithm>
#include <cmath>

namespace dressi {

namespace {

constexpr float kWEps = 1e-6f;
constexpr float kDistEps = 1e-6f;

struct ScreenVert {
    float x, y;   // pixels
    float z;      // NDC depth [0,1]
    float inv_w;  // 1 / clip.w (perspective correction)
};

ScreenVert ProjectToScreen(const float* clip, ImgSize screen) {
    const float inv_w = 1.f / clip[3];
    ScreenVert v;
    v.x = (clip[0] * inv_w * 0.5f + 0.5f) * float(screen.w);
    v.y = (clip[1] * inv_w * 0.5f + 0.5f) * float(screen.h);
    v.z = clip[2] * inv_w;
    v.inv_w = inv_w;
    return v;
}

// Rasterizes each face's screen bbox and calls
// shade(x, y, face, lambda[3]) for covered, depth-passing pixels.
template <typename ShadeFn>
void RasterizeCore(const CpuTensor& clip, const CpuTensor& faces,
                   ImgSize screen, const ShadeFn& shade) {
    DRESSI_CHECK(clip.vtype == VEC4 && clip.size.h == 1,
                 "CPU raster: clip must be VEC4 {V,1}");
    DRESSI_CHECK(faces.numComp() == 3 && faces.size.h == 1,
                 "CPU raster: faces must have 3 components {F,1}");
    const uint32_t n_verts = clip.size.w;
    const uint32_t n_faces = faces.size.w;

    std::vector<float> zbuf(size_t(screen.w) * screen.h, 1.f);

    for (uint32_t f = 0; f < n_faces; f++) {
        uint32_t idx[3];
        ScreenVert sv[3];
        bool skip = false;
        for (int k = 0; k < 3; k++) {
            const float fi = faces.data[size_t(f) * 3 + k];
            idx[k] = uint32_t(int64_t(fi));
            DRESSI_CHECK(idx[k] < n_verts, "CPU raster: face index range");
            const float* c = &clip.data[size_t(idx[k]) * 4];
            if (std::abs(c[3]) <= kWEps || c[3] < 0.f) {
                skip = true;  // no clipping support: reject behind-camera
                break;
            }
            sv[k] = ProjectToScreen(c, screen);
        }
        if (skip) {
            continue;
        }

        const float area = (sv[1].x - sv[0].x) * (sv[2].y - sv[0].y) -
                           (sv[1].y - sv[0].y) * (sv[2].x - sv[0].x);
        if (std::abs(area) < kDistEps) {
            continue;  // degenerate
        }
        const float inv_area = 1.f / area;

        const float min_xf = std::min({sv[0].x, sv[1].x, sv[2].x});
        const float max_xf = std::max({sv[0].x, sv[1].x, sv[2].x});
        const float min_yf = std::min({sv[0].y, sv[1].y, sv[2].y});
        const float max_yf = std::max({sv[0].y, sv[1].y, sv[2].y});
        const int min_x = std::max(0, int(std::floor(min_xf - 0.5f)));
        const int max_x = std::min(int(screen.w) - 1,
                                   int(std::ceil(max_xf - 0.5f)));
        const int min_y = std::max(0, int(std::floor(min_yf - 0.5f)));
        const int max_y = std::min(int(screen.h) - 1,
                                   int(std::ceil(max_yf - 0.5f)));

        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                const float px = float(x) + 0.5f;
                const float py = float(y) + 0.5f;
                // Screen barycentrics (sign-normalized by the area)
                const float l0 = ((sv[1].x - px) * (sv[2].y - py) -
                                  (sv[1].y - py) * (sv[2].x - px)) *
                                 inv_area;
                const float l1 = ((sv[2].x - px) * (sv[0].y - py) -
                                  (sv[2].y - py) * (sv[0].x - px)) *
                                 inv_area;
                const float l2 = 1.f - l0 - l1;
                if (l0 < 0.f || l1 < 0.f || l2 < 0.f) {
                    continue;
                }
                // NDC z is affine in screen space
                const float z = l0 * sv[0].z + l1 * sv[1].z + l2 * sv[2].z;
                if (z < 0.f || z > 1.f) {
                    continue;
                }
                float& zb = zbuf[size_t(y) * screen.w + x];
                if (z > zb) {
                    continue;  // LessOrEqual keeps the later face on ties
                }
                zb = z;
                const float lambda[3] = {l0, l1, l2};
                shade(uint32_t(x), uint32_t(y), f, idx, sv, lambda);
            }
        }
    }
}

}  // namespace

TriDist SignedDistToTri(const float p[2], const float s[3][2]) {
    TriDist best;
    best.dist = 0.f;
    float best_d2 = -1.f;
    float best_q[2] = {0.f, 0.f};
    int n_pos = 0, n_neg = 0;
    for (int k = 0; k < 3; k++) {
        const float* a = s[k];
        const float* b = s[(k + 1) % 3];
        const float ex = b[0] - a[0];
        const float ey = b[1] - a[1];
        const float len2 = ex * ex + ey * ey;
        float t = 0.f;
        if (len2 > kDistEps * kDistEps) {
            t = ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / len2;
            t = std::min(std::max(t, 0.f), 1.f);
        }
        const float qx = a[0] + t * ex;
        const float qy = a[1] + t * ey;
        const float dx = p[0] - qx;
        const float dy = p[1] - qy;
        const float d2 = dx * dx + dy * dy;
        if (best_d2 < 0.f || d2 < best_d2) {
            best_d2 = d2;
            best.edge = k;
            best.t = t;
            best_q[0] = qx;
            best_q[1] = qy;
        }
        const float cr = ex * (p[1] - a[1]) - ey * (p[0] - a[0]);
        n_pos += (cr >= 0.f);
        n_neg += (cr <= 0.f);
    }
    const float sgn = (n_pos == 3 || n_neg == 3) ? 1.f : -1.f;
    best.dist = sgn * std::sqrt(std::max(best_d2, 0.f));
    (void)best_q;
    return best;
}

void SignedDistGradToTri(const float p[2], const float s[3][2],
                         const TriDist& td, float g_s[3][2]) {
    for (int k = 0; k < 3; k++) {
        g_s[k][0] = 0.f;
        g_s[k][1] = 0.f;
    }
    const float d = std::abs(td.dist);
    if (d < kDistEps) {
        return;  // on the boundary: subgradient 0
    }
    const int ia = td.edge;
    const int ib = (td.edge + 1) % 3;
    const float qx = s[ia][0] + td.t * (s[ib][0] - s[ia][0]);
    const float qy = s[ia][1] + td.t * (s[ib][1] - s[ia][1]);
    const float sgn = td.dist >= 0.f ? 1.f : -1.f;
    const float nx = (p[0] - qx) / d;
    const float ny = (p[1] - qy) / d;
    // d|p-q|/da = -(1-t) n, d|p-q|/db = -t n (t held fixed: envelope theorem)
    g_s[ia][0] = -sgn * (1.f - td.t) * nx;
    g_s[ia][1] = -sgn * (1.f - td.t) * ny;
    g_s[ib][0] = -sgn * td.t * nx;
    g_s[ib][1] = -sgn * td.t * ny;
}

bool ScreenBarycentric(const float p[2], const float s[3][2],
                       float lambda[3]) {
    const float area = (s[1][0] - s[0][0]) * (s[2][1] - s[0][1]) -
                       (s[1][1] - s[0][1]) * (s[2][0] - s[0][0]);
    if (std::abs(area) < kDistEps) {
        return false;
    }
    const float inv_area = 1.f / area;
    lambda[0] = ((s[1][0] - p[0]) * (s[2][1] - p[1]) -
                 (s[1][1] - p[1]) * (s[2][0] - p[0])) *
                inv_area;
    lambda[1] = ((s[2][0] - p[0]) * (s[0][1] - p[1]) -
                 (s[2][1] - p[1]) * (s[0][0] - p[0])) *
                inv_area;
    lambda[2] = 1.f - lambda[0] - lambda[1];
    return true;
}

CpuTensor RasterizeHardCpu(const CpuTensor& clip, const CpuTensor& attrib,
                           const CpuTensor& faces, ImgSize screen) {
    const uint32_t n_comp = attrib.numComp();
    DRESSI_CHECK(attrib.size == clip.size,
                 "CPU raster: attribute count must match vertex count");
    CpuTensor out;
    out.vtype = attrib.vtype;
    out.size = screen;
    out.data.assign(size_t(screen.w) * screen.h * n_comp, 0.f);

    RasterizeCore(clip, faces, screen,
                  [&](uint32_t x, uint32_t y, uint32_t /*f*/,
                      const uint32_t idx[3], const ScreenVert sv[3],
                      const float lambda[3]) {
                      // Perspective-correct: sum(l*a/w) / sum(l/w)
                      const float denom = lambda[0] * sv[0].inv_w +
                                          lambda[1] * sv[1].inv_w +
                                          lambda[2] * sv[2].inv_w;
                      float* dst =
                              &out.data[(size_t(y) * screen.w + x) * n_comp];
                      for (uint32_t c = 0; c < n_comp; c++) {
                          float num = 0.f;
                          for (int k = 0; k < 3; k++) {
                              num += lambda[k] * sv[k].inv_w *
                                     attrib.data[size_t(idx[k]) * n_comp + c];
                          }
                          dst[c] = num / denom;
                      }
                  });
    return out;
}

CpuTensor RasterizeFaceIdCpu(const CpuTensor& clip, const CpuTensor& faces,
                             ImgSize screen) {
    CpuTensor out;
    out.vtype = FLOAT;
    out.size = screen;
    out.data.assign(size_t(screen.w) * screen.h, 0.f);
    RasterizeCore(clip, faces, screen,
                  [&](uint32_t x, uint32_t y, uint32_t f, const uint32_t*,
                      const ScreenVert*, const float*) {
                      out.data[size_t(y) * screen.w + x] = float(f) + 1.f;
                  });
    return out;
}

}  // namespace dressi
