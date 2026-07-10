#include "dressi/mesh_utils.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace dressi {

std::vector<std::vector<uint32_t>> BuildVertexAdjacency(const CpuImage& faces,
                                                        uint32_t n_verts) {
    std::vector<std::set<uint32_t>> adj_sets(n_verts);
    for (uint32_t f = 0; f < faces.width; f++) {
        uint32_t idx[3];
        for (int k = 0; k < 3; k++) {
            idx[k] = uint32_t(faces.at(f, 0, uint32_t(k)));
        }
        for (int k = 0; k < 3; k++) {
            adj_sets[idx[k]].insert(idx[(k + 1) % 3]);
            adj_sets[idx[k]].insert(idx[(k + 2) % 3]);
        }
    }
    std::vector<std::vector<uint32_t>> adj(n_verts);
    for (uint32_t v = 0; v < n_verts; v++) {
        adj[v].assign(adj_sets[v].begin(), adj_sets[v].end());
    }
    return adj;
}

std::vector<std::array<uint32_t, 2>> BuildFaceAdjacency(
        const CpuImage& faces) {
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> edge_faces;
    for (uint32_t f = 0; f < faces.width; f++) {
        for (int k = 0; k < 3; k++) {
            const uint32_t a = uint32_t(faces.at(f, 0, uint32_t(k)));
            const uint32_t b =
                    uint32_t(faces.at(f, 0, uint32_t((k + 1) % 3)));
            edge_faces[std::minmax(a, b)].push_back(f);
        }
    }
    std::vector<std::array<uint32_t, 2>> pairs;
    for (const auto& [edge, fs] : edge_faces) {
        (void)edge;
        for (size_t i = 0; i < fs.size(); i++) {
            for (size_t j = i + 1; j < fs.size(); j++) {
                pairs.push_back({fs[i], fs[j]});
            }
        }
    }
    return pairs;
}

CpuImage VertexFacesTex(const CpuImage& faces, uint32_t n_verts) {
    std::vector<std::vector<uint32_t>> vf(n_verts);
    for (uint32_t f = 0; f < faces.width; f++) {
        for (int k = 0; k < 3; k++) {
            vf[uint32_t(faces.at(f, 0, uint32_t(k)))].push_back(f);
        }
    }
    uint32_t max_deg = 1;
    for (const auto& fs : vf) {
        max_deg = std::max(max_deg, uint32_t(fs.size()));
    }
    CpuImage tex(n_verts, max_deg, 1, -1.f);
    for (uint32_t v = 0; v < n_verts; v++) {
        for (uint32_t d = 0; d < vf[v].size(); d++) {
            tex.at(v, d, 0) = float(vf[v][d]);
        }
    }
    return tex;
}

CpuImage VertexNeighborsTex(const CpuImage& faces, uint32_t n_verts) {
    const auto adj = BuildVertexAdjacency(faces, n_verts);
    uint32_t max_deg = 1;
    for (const auto& ns : adj) {
        max_deg = std::max(max_deg, uint32_t(ns.size()));
    }
    CpuImage tex(n_verts, max_deg, 1, -1.f);
    for (uint32_t v = 0; v < n_verts; v++) {
        for (uint32_t d = 0; d < adj[v].size(); d++) {
            tex.at(v, d, 0) = float(adj[v][d]);
        }
    }
    return tex;
}

CpuImage FaceNeighborsTex(const CpuImage& faces) {
    CpuImage tex(faces.width, 1, 3, -1.f);
    std::vector<uint32_t> n_adj(faces.width, 0);
    for (const auto& pair : BuildFaceAdjacency(faces)) {
        for (int e = 0; e < 2; e++) {
            const uint32_t f = pair[size_t(e)];
            const uint32_t g = pair[size_t(1 - e)];
            if (n_adj[f] < 3) {
                tex.at(f, 0, n_adj[f]++) = float(g);
            }
        }
    }
    return tex;
}

SoftGeometry BuildSoftGeometry(const CpuImage& hard_clip,
                               const CpuImage& faces, ImgSize screen,
                               float radius_px) {
    const uint32_t n_faces = faces.width;
    SoftGeometry soft;
    soft.clip = CpuImage(n_faces * 3, 1, 4);
    soft.face_id = CpuImage(n_faces * 3, 1, 1);
    soft.faces = CpuImage(n_faces, 1, 3);

    constexpr float kWEps = 1e-6f;
    constexpr float kMaxScale = 8.f;

    for (uint32_t f = 0; f < n_faces; f++) {
        uint32_t idx[3];
        float clip[3][4];
        float sx[3], sy[3];
        bool valid = true;
        for (int k = 0; k < 3; k++) {
            idx[k] = uint32_t(faces.at(f, 0, uint32_t(k)));
            for (uint32_t c = 0; c < 4; c++) {
                clip[k][c] = hard_clip.at(idx[k], 0, c);
            }
            if (clip[k][3] <= kWEps) {
                valid = false;
            } else {
                sx[k] = (clip[k][0] / clip[k][3] * 0.5f + 0.5f) *
                        float(screen.w);
                sy[k] = (clip[k][1] / clip[k][3] * 0.5f + 0.5f) *
                        float(screen.h);
            }
        }

        float scale = 1.f;
        if (valid) {
            // Min distance from the centroid to an edge; scaling about the
            // centroid by 1 + r/d_min dilates every edge outward by >= r
            const float cx = (sx[0] + sx[1] + sx[2]) / 3.f;
            const float cy = (sy[0] + sy[1] + sy[2]) / 3.f;
            float d_min = 1e30f;
            for (int k = 0; k < 3; k++) {
                const int k2 = (k + 1) % 3;
                const float ex = sx[k2] - sx[k];
                const float ey = sy[k2] - sy[k];
                const float len = std::sqrt(ex * ex + ey * ey);
                if (len < 1e-6f) {
                    continue;
                }
                const float d = std::abs(ex * (cy - sy[k]) -
                                         ey * (cx - sx[k])) /
                                len;
                d_min = std::min(d_min, d);
            }
            if (d_min < 1e29f) {
                scale = std::min(1.f + radius_px / std::max(d_min, 1e-3f),
                                 kMaxScale);
            }
            for (int k = 0; k < 3; k++) {
                const float ex = cx + (sx[k] - cx) * scale;
                const float ey = cy + (sy[k] - cy) * scale;
                // Unproject keeping z and w fixed
                clip[k][0] = (ex / float(screen.w) * 2.f - 1.f) * clip[k][3];
                clip[k][1] = (ey / float(screen.h) * 2.f - 1.f) * clip[k][3];
            }
        }

        for (int k = 0; k < 3; k++) {
            const uint32_t v = f * 3 + uint32_t(k);
            for (uint32_t c = 0; c < 4; c++) {
                soft.clip.at(v, 0, c) = clip[k][c];
            }
            soft.face_id.at(v, 0, 0) = float(f);
            soft.faces.at(f, 0, uint32_t(k)) = float(v);
        }
    }
    return soft;
}

CpuImage UvAsClip(const CpuImage& uv) {
    CpuImage clip(uv.width, 1, 4);
    for (uint32_t i = 0; i < uv.width; i++) {
        clip.at(i, 0, 0) = uv.at(i, 0, 0) * 2.f - 1.f;
        clip.at(i, 0, 1) = uv.at(i, 0, 1) * 2.f - 1.f;
        clip.at(i, 0, 2) = 0.5f;
        clip.at(i, 0, 3) = 1.f;
    }
    return clip;
}

CpuImage ScreenAttr(const CpuImage& clip, ImgSize screen) {
    CpuImage attr(clip.width, 1, 4);
    for (uint32_t i = 0; i < clip.width; i++) {
        const float w = clip.at(i, 0, 3);
        const float ndc_x = clip.at(i, 0, 0) / w;
        const float ndc_y = clip.at(i, 0, 1) / w;
        attr.at(i, 0, 0) = (ndc_x * 0.5f + 0.5f) * float(screen.w);
        attr.at(i, 0, 1) = (ndc_y * 0.5f + 0.5f) * float(screen.h);
        attr.at(i, 0, 2) = 1.f;
        attr.at(i, 0, 3) = 0.f;
    }
    return attr;
}

}  // namespace dressi
