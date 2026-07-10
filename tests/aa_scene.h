// Shared test fixture for the screen-space AA technique: a hard-rasterized
// mask + triangle-ID buffer + AntiAlias over one clip-position leaf (the
// same Variable feeds the rasterizer vertex buffer and the AA texture).

#ifndef DRESSI_TESTS_AA_SCENE_H
#define DRESSI_TESTS_AA_SCENE_H

#include <array>
#include <vector>

#include "core/cpu_eval.h"
#include "dressi/dressi.h"

namespace aa_test {

inline dressi::CpuTensor MakeTensor(dressi::VType vtype, dressi::ImgSize size,
                                    std::vector<float> data) {
    dressi::CpuTensor t;
    t.vtype = vtype;
    t.size = size;
    t.data = std::move(data);
    return t;
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

struct AaScene {
    dressi::Variable clip, ones, faces_i, faces_f, vtx_faces;
    dressi::CpuTensor t_clip, t_ones, t_faces_i, t_faces_f, t_vtx_faces;

    AaScene(const std::vector<std::array<float, 4>>& verts,
            const std::vector<std::array<uint32_t, 3>>& faces)
        : clip(dressi::VEC4, {uint32_t(verts.size()), 1}),
          ones(dressi::FLOAT, {uint32_t(verts.size()), 1}),
          faces_i(dressi::IVEC3, {uint32_t(faces.size()), 1}),
          faces_f(dressi::VEC3, {uint32_t(faces.size()), 1}),
          vtx_faces(nullptr) {
        std::vector<float> vc;
        for (const auto& v : verts) {
            vc.insert(vc.end(), v.begin(), v.end());
        }
        t_clip = MakeTensor(dressi::VEC4, {uint32_t(verts.size()), 1},
                            std::move(vc));
        t_ones = MakeTensor(dressi::FLOAT, {uint32_t(verts.size()), 1},
                            std::vector<float>(verts.size(), 1.f));
        std::vector<float> fc;
        for (const auto& f : faces) {
            fc.insert(fc.end(), {float(f[0]), float(f[1]), float(f[2])});
        }
        t_faces_i = MakeTensor(dressi::IVEC3, {uint32_t(faces.size()), 1},
                               fc);
        t_faces_f = MakeTensor(dressi::VEC3, {uint32_t(faces.size()), 1},
                               std::move(fc));
        t_vtx_faces = VertexFacesTensor(faces, uint32_t(verts.size()));
        vtx_faces = dressi::Variable(dressi::FLOAT, t_vtx_faces.size);
    }

    // mask/tri/AA graph over the shared clip leaf
    dressi::Variable antialiasedMask(dressi::ImgSize screen) const {
        using namespace dressi;
        Variable mask = F::Rasterize(clip, ones, faces_i, screen);
        Variable tri = F::RasterizeFaceId(clip, ones, faces_i, screen);
        return F::AntiAlias(mask, tri, clip, faces_f, vtx_faces);
    }

    void bind(dressi::CpuEvaluator& ev) const {
        ev.bind(clip, t_clip);
        ev.bind(ones, t_ones);
        ev.bind(faces_i, t_faces_i);
        ev.bind(faces_f, t_faces_f);
        ev.bind(vtx_faces, t_vtx_faces);
    }

    void send(dressi::DressiAD& ad) const {
        ad.sendImg(clip, dressi::CpuImageFromTensor(t_clip));
        ad.sendImg(ones, dressi::CpuImageFromTensor(t_ones));
        ad.sendImg(faces_i, dressi::CpuImageFromTensor(t_faces_i));
        ad.sendImg(faces_f, dressi::CpuImageFromTensor(t_faces_f));
        ad.sendImg(vtx_faces, dressi::CpuImageFromTensor(t_vtx_faces));
    }
};

}  // namespace aa_test

#endif  // DRESSI_TESTS_AA_SCENE_H
