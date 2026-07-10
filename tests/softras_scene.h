// Shared test fixture: a HardSoftRas scene with CPU-enlarged soft
// triangles (screen-space centroid scaling; per-face w must be uniform for
// the clip-space shortcut used here).

#ifndef DRESSI_TESTS_SOFTRAS_SCENE_H
#define DRESSI_TESTS_SOFTRAS_SCENE_H

#include <array>
#include <vector>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"

namespace softras_test {

inline dressi::CpuTensor MakeTensor(dressi::VType vtype, dressi::ImgSize size,
                                    std::vector<float> data) {
    dressi::CpuTensor t;
    t.vtype = vtype;
    t.size = size;
    t.data = std::move(data);
    return t;
}

// Clip position for a screen-space point: clip = (s / S * 2 - 1) * w
inline std::array<float, 4> ClipOfScreen(float sx, float sy, float z,
                                         uint32_t screen, float w = 1.f) {
    return {(sx / float(screen) * 2.f - 1.f) * w,
            (sy / float(screen) * 2.f - 1.f) * w, z * w, w};
}

// Vertex -> incident-face adjacency tensor {V, max_deg} (-1 padding)
inline dressi::CpuTensor VertexFacesTensor(
        const std::vector<std::array<uint32_t, 3>>& faces, uint32_t n_verts) {
    std::vector<std::vector<uint32_t>> vf(n_verts);
    for (uint32_t f = 0; f < faces.size(); f++) {
        for (int k = 0; k < 3; k++) {
            vf[faces[f][size_t(k)]].push_back(f);
        }
    }
    uint32_t max_deg = 1;
    for (const auto& fs : vf) {
        max_deg = std::max(max_deg, uint32_t(fs.size()));
    }
    dressi::CpuTensor t;
    t.vtype = dressi::FLOAT;
    t.size = {n_verts, max_deg};
    t.data.assign(size_t(n_verts) * max_deg, -1.f);
    for (uint32_t v = 0; v < n_verts; v++) {
        for (uint32_t d = 0; d < vf[v].size(); d++) {
            t.data[size_t(d) * n_verts + v] = float(vf[v][d]);
        }
    }
    return t;
}

struct SoftScene {
    dressi::Variable soft_clip, face_id, faces_soft, hard_clip, faces_tex,
            vtx_faces;
    dressi::CpuTensor t_soft_clip, t_face_id, t_faces_soft, t_hard_clip,
            t_faces_tex, t_vtx_faces;

    SoftScene(const std::vector<std::array<float, 4>>& hard_verts,
              const std::vector<std::array<uint32_t, 3>>& faces, float scale)
        : soft_clip(dressi::VEC4, {uint32_t(faces.size()) * 3, 1}),
          face_id(dressi::FLOAT, {uint32_t(faces.size()) * 3, 1}),
          faces_soft(dressi::IVEC3, {uint32_t(faces.size()), 1}),
          hard_clip(dressi::VEC4, {uint32_t(hard_verts.size()), 1}),
          faces_tex(dressi::VEC3, {uint32_t(faces.size()), 1}),
          vtx_faces(nullptr) {
        t_vtx_faces =
                VertexFacesTensor(faces, uint32_t(hard_verts.size()));
        vtx_faces = dressi::Variable(dressi::FLOAT, t_vtx_faces.size);
        const uint32_t n_faces = uint32_t(faces.size());
        t_face_id = MakeTensor(dressi::FLOAT, {n_faces * 3, 1},
                               std::vector<float>(n_faces * 3, 0.f));
        t_faces_soft = MakeTensor(dressi::IVEC3, {n_faces, 1},
                                  std::vector<float>(n_faces * 3, 0.f));
        t_faces_tex = MakeTensor(dressi::VEC3, {n_faces, 1},
                                 std::vector<float>(n_faces * 3, 0.f));
        std::vector<float> hc;
        for (const auto& v : hard_verts) {
            hc.insert(hc.end(), v.begin(), v.end());
        }
        t_hard_clip = MakeTensor(dressi::VEC4,
                                 {uint32_t(hard_verts.size()), 1},
                                 std::move(hc));
        for (uint32_t f = 0; f < n_faces; f++) {
            for (int k = 0; k < 3; k++) {
                t_face_id.data[f * 3 + uint32_t(k)] = float(f);
                t_faces_soft.data[f * 3 + uint32_t(k)] =
                        float(f * 3 + uint32_t(k));
                t_faces_tex.data[f * 3 + uint32_t(k)] =
                        float(faces[f][size_t(k)]);
            }
        }
        m_faces = faces;
        m_scale = scale;
        rebuildSoft();
    }

    // Re-derives the enlarged soft clip positions from t_hard_clip (call
    // after mutating the hard positions during optimization tests)
    void rebuildSoft() {
        const uint32_t n_faces = uint32_t(m_faces.size());
        std::vector<float> sc(size_t(n_faces) * 12, 0.f);
        for (uint32_t f = 0; f < n_faces; f++) {
            float cx = 0.f, cy = 0.f;
            for (int k = 0; k < 3; k++) {
                const float* v =
                        &t_hard_clip.data[size_t(m_faces[f][size_t(k)]) * 4];
                cx += v[0] / v[3] / 3.f;
                cy += v[1] / v[3] / 3.f;
            }
            for (int k = 0; k < 3; k++) {
                const float* v =
                        &t_hard_clip.data[size_t(m_faces[f][size_t(k)]) * 4];
                const float w = v[3];
                float* dst = &sc[(size_t(f) * 3 + size_t(k)) * 4];
                dst[0] = (cx + (v[0] / w - cx) * m_scale) * w;
                dst[1] = (cy + (v[1] / w - cy) * m_scale) * w;
                dst[2] = v[2];
                dst[3] = w;
            }
        }
        t_soft_clip = MakeTensor(dressi::VEC4, {n_faces * 3, 1},
                                 std::move(sc));
    }

    void bind(dressi::CpuEvaluator& ev) const {
        ev.bind(soft_clip, t_soft_clip);
        ev.bind(face_id, t_face_id);
        ev.bind(faces_soft, t_faces_soft);
        ev.bind(hard_clip, t_hard_clip);
        ev.bind(faces_tex, t_faces_tex);
        ev.bind(vtx_faces, t_vtx_faces);
    }

    void send(dressi::DressiAD& ad) const {
        ad.sendImg(soft_clip, dressi::CpuImageFromTensor(t_soft_clip));
        ad.sendImg(face_id, dressi::CpuImageFromTensor(t_face_id));
        ad.sendImg(faces_soft, dressi::CpuImageFromTensor(t_faces_soft));
        ad.sendImg(hard_clip, dressi::CpuImageFromTensor(t_hard_clip));
        ad.sendImg(faces_tex, dressi::CpuImageFromTensor(t_faces_tex));
        ad.sendImg(vtx_faces, dressi::CpuImageFromTensor(t_vtx_faces));
    }

    dressi::Variable rasterize(dressi::ImgSize screen, float radius) const {
        return dressi::F::RasterizeSoft(soft_clip, face_id, faces_soft,
                                        hard_clip, faces_tex, vtx_faces,
                                        screen, radius);
    }

private:
    std::vector<std::array<uint32_t, 3>> m_faces;
    float m_scale = 1.f;
};

// pred = coverage * sigmoid(dist / (radius / 7))
inline dressi::Variable SoftSilhouette(const dressi::Variable& raster_out,
                                       float radius) {
    using namespace dressi;
    return F::StopGradient(F::GetW(raster_out)) *
           F::Sigmoid(F::GetX(raster_out) * (7.f / radius));
}

}  // namespace softras_test

#endif  // DRESSI_TESTS_SOFTRAS_SCENE_H
