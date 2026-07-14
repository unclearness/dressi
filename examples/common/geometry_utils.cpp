#include "geometry_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>

namespace dressi_examples {

namespace {

struct V3 {
    float x, y, z;
};

V3 Normalized(const V3& v) {
    const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x / len, v.y / len, v.z / len};
}

}  // namespace

Mesh GenerateIcosphere(uint32_t level) {
    // Icosahedron from three orthogonal golden rectangles
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<V3> verts = {
            {-1, t, 0}, {1, t, 0},   {-1, -t, 0}, {1, -t, 0},
            {0, -1, t}, {0, 1, t},   {0, -1, -t}, {0, 1, -t},
            {t, 0, -1}, {t, 0, 1},   {-t, 0, -1}, {-t, 0, 1}};
    for (auto& v : verts) {
        v = Normalized(v);
    }
    std::vector<std::array<uint32_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1},   {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
            {1, 5, 9},  {5, 11, 4},  {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4},  {3, 4, 2},   {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
            {4, 9, 5},  {2, 4, 11},  {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};

    for (uint32_t l = 0; l < level; l++) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoint;
        const auto mid = [&](uint32_t a, uint32_t b) -> uint32_t {
            const auto key = std::minmax(a, b);
            if (auto it = midpoint.find(key); it != midpoint.end()) {
                return it->second;
            }
            const V3 m = Normalized({(verts[a].x + verts[b].x) * 0.5f,
                                     (verts[a].y + verts[b].y) * 0.5f,
                                     (verts[a].z + verts[b].z) * 0.5f});
            verts.push_back(m);
            const uint32_t idx = uint32_t(verts.size() - 1);
            midpoint[key] = idx;
            return idx;
        };
        std::vector<std::array<uint32_t, 3>> next;
        next.reserve(faces.size() * 4);
        for (const auto& f : faces) {
            const uint32_t a = mid(f[0], f[1]);
            const uint32_t b = mid(f[1], f[2]);
            const uint32_t c = mid(f[2], f[0]);
            next.push_back({f[0], a, c});
            next.push_back({f[1], b, a});
            next.push_back({f[2], c, b});
            next.push_back({a, b, c});
        }
        faces = std::move(next);
    }

    Mesh mesh;
    mesh.pos = dressi::CpuImage(uint32_t(verts.size()), 1, 3);
    mesh.uv = dressi::CpuImage(uint32_t(verts.size()), 1, 2, 0.f);
    mesh.faces = dressi::CpuImage(uint32_t(faces.size()), 1, 3);
    for (uint32_t i = 0; i < verts.size(); i++) {
        mesh.pos.at(i, 0, 0) = verts[i].x;
        mesh.pos.at(i, 0, 1) = verts[i].y;
        mesh.pos.at(i, 0, 2) = verts[i].z;
    }
    for (uint32_t i = 0; i < faces.size(); i++) {
        for (uint32_t c = 0; c < 3; c++) {
            mesh.faces.at(i, 0, c) = float(faces[i][c]);
        }
    }
    return mesh;
}

Mesh GenerateUvSphere(uint32_t stacks, uint32_t slices) {
    stacks = std::max(2u, stacks);
    slices = std::max(3u, slices);
    const uint32_t cols = slices + 1;  // duplicate the seam column
    const float kPi = 3.14159265358979323846f;
    Mesh mesh;
    mesh.pos = dressi::CpuImage(cols * (stacks + 1), 1, 3);
    mesh.uv = dressi::CpuImage(cols * (stacks + 1), 1, 2);
    const auto vid = [&](uint32_t i, uint32_t j) { return i * cols + j; };
    for (uint32_t i = 0; i <= stacks; i++) {
        const float theta = kPi * float(i) / float(stacks);  // 0=top..pi
        const float st = std::sin(theta), ct = std::cos(theta);
        for (uint32_t j = 0; j <= slices; j++) {
            const float phi = 2.f * kPi * float(j) / float(slices);
            const uint32_t v = vid(i, j);
            mesh.pos.at(v, 0, 0) = st * std::cos(phi);
            mesh.pos.at(v, 0, 1) = ct;
            mesh.pos.at(v, 0, 2) = st * std::sin(phi);
            mesh.uv.at(v, 0, 0) = float(j) / float(slices);
            mesh.uv.at(v, 0, 1) = float(i) / float(stacks);
        }
    }
    std::vector<std::array<uint32_t, 3>> faces;
    faces.reserve(size_t(stacks) * slices * 2);
    for (uint32_t i = 0; i < stacks; i++) {
        for (uint32_t j = 0; j < slices; j++) {
            const uint32_t tl = vid(i, j), tr = vid(i, j + 1);
            const uint32_t bl = vid(i + 1, j), br = vid(i + 1, j + 1);
            // Skip the degenerate pole triangle (tl==tr at the top cap,
            // bl==br at the bottom cap); winding is CCW seen from OUTSIDE.
            if (i != stacks - 1) {
                faces.push_back({tl, br, bl});
            }
            if (i != 0) {
                faces.push_back({tl, tr, br});
            }
        }
    }
    mesh.faces = dressi::CpuImage(uint32_t(faces.size()), 1, 3);
    for (uint32_t i = 0; i < faces.size(); i++) {
        for (uint32_t c = 0; c < 3; c++) {
            mesh.faces.at(i, 0, c) = float(faces[i][c]);
        }
    }
    return mesh;
}

dressi::CpuImage TransferDeformationByDirection(
        const dressi::CpuImage& ctrl_rest_pos,
        const dressi::CpuImage& ctrl_faces,
        const dressi::CpuImage& ctrl_deformed_pos,
        const dressi::CpuImage& dst_pos) {
    const auto rest = [&](uint32_t v, uint32_t c) {
        return ctrl_rest_pos.at(v, 0, c);
    };
    dressi::CpuImage out(dst_pos.width, 1, 3);
    const uint32_t n_faces = ctrl_faces.width;
    for (uint32_t d = 0; d < dst_pos.width; d++) {
        const V3 dir = Normalized({dst_pos.at(d, 0, 0), dst_pos.at(d, 0, 1),
                                   dst_pos.at(d, 0, 2)});
        float best_t = 1e30f;
        uint32_t hit = 0;
        float hu = 0.f, hv = 0.f;
        for (uint32_t f = 0; f < n_faces; f++) {
            const uint32_t ia = uint32_t(ctrl_faces.at(f, 0, 0));
            const uint32_t ib = uint32_t(ctrl_faces.at(f, 0, 1));
            const uint32_t ic = uint32_t(ctrl_faces.at(f, 0, 2));
            // Moeller-Trumbore, ray origin = 0, direction = dir
            const V3 e1 = {rest(ib, 0) - rest(ia, 0), rest(ib, 1) - rest(ia, 1),
                           rest(ib, 2) - rest(ia, 2)};
            const V3 e2 = {rest(ic, 0) - rest(ia, 0), rest(ic, 1) - rest(ia, 1),
                           rest(ic, 2) - rest(ia, 2)};
            const V3 p = {dir.y * e2.z - dir.z * e2.y,
                          dir.z * e2.x - dir.x * e2.z,
                          dir.x * e2.y - dir.y * e2.x};
            const float det = e1.x * p.x + e1.y * p.y + e1.z * p.z;
            if (std::abs(det) < 1e-20f) {
                continue;
            }
            const float inv = 1.f / det;
            const V3 tv = {-rest(ia, 0), -rest(ia, 1), -rest(ia, 2)};
            const float u = (tv.x * p.x + tv.y * p.y + tv.z * p.z) * inv;
            if (u < -1e-4f || u > 1.f + 1e-4f) {
                continue;
            }
            const V3 q = {tv.y * e1.z - tv.z * e1.y, tv.z * e1.x - tv.x * e1.z,
                          tv.x * e1.y - tv.y * e1.x};
            const float v = (dir.x * q.x + dir.y * q.y + dir.z * q.z) * inv;
            if (v < -1e-4f || u + v > 1.f + 1e-4f) {
                continue;
            }
            const float t = (e2.x * q.x + e2.y * q.y + e2.z * q.z) * inv;
            if (t > 1e-6f && t < best_t) {
                best_t = t;
                hit = f;
                hu = u;
                hv = v;
            }
        }
        // point = a + u*(b-a) + v*(c-a) -> weights (1-u-v, u, v) on a,b,c
        const uint32_t ia = uint32_t(ctrl_faces.at(hit, 0, 0));
        const uint32_t ib = uint32_t(ctrl_faces.at(hit, 0, 1));
        const uint32_t ic = uint32_t(ctrl_faces.at(hit, 0, 2));
        const float wa = 1.f - hu - hv;
        for (uint32_t c = 0; c < 3; c++) {
            out.at(d, 0, c) = wa * ctrl_deformed_pos.at(ia, 0, c) +
                              hu * ctrl_deformed_pos.at(ib, 0, c) +
                              hv * ctrl_deformed_pos.at(ic, 0, c);
        }
    }
    return out;
}

void SaveObjMesh(const std::string& path, const Mesh& mesh) {
    std::ofstream ofs(path);
    if (!ofs) {
        return;
    }
    for (uint32_t i = 0; i < mesh.numVertices(); i++) {
        ofs << "v " << mesh.pos.at(i, 0, 0) << ' ' << mesh.pos.at(i, 0, 1)
            << ' ' << mesh.pos.at(i, 0, 2) << '\n';
    }
    for (uint32_t i = 0; i < mesh.numFaces(); i++) {
        ofs << "f " << int(mesh.faces.at(i, 0, 0)) + 1 << ' '
            << int(mesh.faces.at(i, 0, 1)) + 1 << ' '
            << int(mesh.faces.at(i, 0, 2)) + 1 << '\n';
    }
}

void SaveTexturedObjMesh(const std::string& path, const Mesh& mesh,
                         const std::string& texture_filename) {
    std::ofstream ofs(path);
    if (!ofs) {
        return;
    }
    // Companion .mtl (same stem as the OBJ) referencing the texture
    std::string mtl_path = path;
    const size_t dot = mtl_path.find_last_of('.');
    if (dot != std::string::npos) {
        mtl_path = mtl_path.substr(0, dot);
    }
    mtl_path += ".mtl";
    std::string mtl_name = mtl_path;
    if (const size_t slash = mtl_name.find_last_of("/\\");
        slash != std::string::npos) {
        mtl_name = mtl_name.substr(slash + 1);
    }
    if (std::ofstream mtl(mtl_path); mtl) {
        mtl << "newmtl material0\n"
            << "Ka 0 0 0\nKd 1 1 1\nd 1\nillum 1\n"
            << "map_Kd " << texture_filename << '\n';
    }

    ofs << "mtllib " << mtl_name << "\nusemtl material0\n";
    for (uint32_t i = 0; i < mesh.numVertices(); i++) {
        ofs << "v " << mesh.pos.at(i, 0, 0) << ' ' << mesh.pos.at(i, 0, 1)
            << ' ' << mesh.pos.at(i, 0, 2) << '\n';
    }
    for (uint32_t i = 0; i < mesh.numVertices(); i++) {
        // OBJ/GL sampler origin is bottom-left; dressi uv is top-left.
        ofs << "vt " << mesh.uv.at(i, 0, 0) << ' '
            << 1.f - mesh.uv.at(i, 0, 1) << '\n';
    }
    for (uint32_t i = 0; i < mesh.numFaces(); i++) {
        const int a = int(mesh.faces.at(i, 0, 0)) + 1;
        const int b = int(mesh.faces.at(i, 0, 1)) + 1;
        const int c = int(mesh.faces.at(i, 0, 2)) + 1;
        ofs << "f " << a << '/' << a << ' ' << b << '/' << b << ' ' << c
            << '/' << c << '\n';
    }
}

dressi::CpuImage UniformLaplacianGrad(
        const dressi::CpuImage& pos,
        const std::vector<std::vector<uint32_t>>& adj, float lambda) {
    dressi::CpuImage grad(pos.width, 1, 3, 0.f);
    for (uint32_t v = 0; v < pos.width; v++) {
        if (adj[v].empty()) {
            continue;
        }
        float mean[3] = {0.f, 0.f, 0.f};
        for (uint32_t n : adj[v]) {
            for (uint32_t c = 0; c < 3; c++) {
                mean[c] += pos.at(n, 0, c);
            }
        }
        const float inv_n = 1.f / float(adj[v].size());
        for (uint32_t c = 0; c < 3; c++) {
            grad.at(v, 0, c) = lambda * (pos.at(v, 0, c) - mean[c] * inv_n);
        }
    }
    return grad;
}

namespace {

V3 FaceCross(const dressi::CpuImage& pos, const dressi::CpuImage& faces,
             uint32_t f, uint32_t idx[3]) {
    for (int k = 0; k < 3; k++) {
        idx[k] = uint32_t(faces.at(f, 0, uint32_t(k)));
    }
    const auto p = [&](uint32_t v, uint32_t c) { return pos.at(v, 0, c); };
    const V3 e1 = {p(idx[1], 0) - p(idx[0], 0), p(idx[1], 1) - p(idx[0], 1),
                   p(idx[1], 2) - p(idx[0], 2)};
    const V3 e2 = {p(idx[2], 0) - p(idx[0], 0), p(idx[2], 1) - p(idx[0], 1),
                   p(idx[2], 2) - p(idx[0], 2)};
    return {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z,
            e1.x * e2.y - e1.y * e2.x};
}

}  // namespace

dressi::CpuImage NormalConsistencyGrad(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj, float lambda) {
    dressi::CpuImage grad(pos.width, 1, 3, 0.f);
    const auto add_face_grad = [&](uint32_t f, const uint32_t idx[3],
                                   const V3& g) {
        // c = e1 x e2 with e1 = p1 - p0, e2 = p2 - p0:
        // dE via g . dc  =>  grad_p1 += e2 x g, grad_p2 += g x e1,
        // grad_p0 -= both (translation invariance)
        (void)f;
        const auto p = [&](uint32_t v, uint32_t c) {
            return pos.at(v, 0, c);
        };
        const V3 e1 = {p(idx[1], 0) - p(idx[0], 0),
                       p(idx[1], 1) - p(idx[0], 1),
                       p(idx[1], 2) - p(idx[0], 2)};
        const V3 e2 = {p(idx[2], 0) - p(idx[0], 0),
                       p(idx[2], 1) - p(idx[0], 1),
                       p(idx[2], 2) - p(idx[0], 2)};
        const V3 g1 = {e2.y * g.z - e2.z * g.y, e2.z * g.x - e2.x * g.z,
                       e2.x * g.y - e2.y * g.x};  // e2 x g
        const V3 g2 = {g.y * e1.z - g.z * e1.y, g.z * e1.x - g.x * e1.z,
                       g.x * e1.y - g.y * e1.x};  // g x e1
        grad.at(idx[1], 0, 0) += g1.x;
        grad.at(idx[1], 0, 1) += g1.y;
        grad.at(idx[1], 0, 2) += g1.z;
        grad.at(idx[2], 0, 0) += g2.x;
        grad.at(idx[2], 0, 1) += g2.y;
        grad.at(idx[2], 0, 2) += g2.z;
        grad.at(idx[0], 0, 0) -= g1.x + g2.x;
        grad.at(idx[0], 0, 1) -= g1.y + g2.y;
        grad.at(idx[0], 0, 2) -= g1.z + g2.z;
    };

    for (const auto& pair : face_adj) {
        uint32_t ia[3], ib[3];
        const V3 ca = FaceCross(pos, faces, pair[0], ia);
        const V3 cb = FaceCross(pos, faces, pair[1], ib);
        const float la =
                std::sqrt(ca.x * ca.x + ca.y * ca.y + ca.z * ca.z);
        const float lb =
                std::sqrt(cb.x * cb.x + cb.y * cb.y + cb.z * cb.z);
        if (la < 1e-12f || lb < 1e-12f) {
            continue;  // degenerate face: no defined normal
        }
        const V3 na = {ca.x / la, ca.y / la, ca.z / la};
        const V3 nb = {cb.x / lb, cb.y / lb, cb.z / lb};
        const float d = na.x * nb.x + na.y * nb.y + na.z * nb.z;
        // E = lambda (1 - n_a . n_b);
        // dE/dc_a = -lambda (n_b - d n_a) / |c_a|   (and symmetrically)
        const V3 ga = {-lambda * (nb.x - d * na.x) / la,
                       -lambda * (nb.y - d * na.y) / la,
                       -lambda * (nb.z - d * na.z) / la};
        const V3 gb = {-lambda * (na.x - d * nb.x) / lb,
                       -lambda * (na.y - d * nb.y) / lb,
                       -lambda * (na.z - d * nb.z) / lb};
        add_face_grad(pair[0], ia, ga);
        add_face_grad(pair[1], ib, gb);
    }
    return grad;
}

uint32_t CountFlippedFacePairs(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj) {
    uint32_t flipped = 0;
    for (const auto& pair : face_adj) {
        uint32_t ia[3], ib[3];
        const V3 ca = FaceCross(pos, faces, pair[0], ia);
        const V3 cb = FaceCross(pos, faces, pair[1], ib);
        if (ca.x * cb.x + ca.y * cb.y + ca.z * cb.z < 0.f) {
            flipped++;
        }
    }
    return flipped;
}

dressi::CpuImage ChainRuleClipToPos(const dressi::CpuImage& g_clip,
                                    const Mat4& mvp) {
    // clip_c = sum_j mvp[j*4+c] * pos_j + mvp[12+c]  (column-major)
    dressi::CpuImage g_pos(g_clip.width, 1, 3, 0.f);
    for (uint32_t v = 0; v < g_clip.width; v++) {
        for (uint32_t j = 0; j < 3; j++) {
            float g = 0.f;
            for (uint32_t c = 0; c < 4; c++) {
                g += g_clip.at(v, 0, c) * mvp[j * 4 + c];
            }
            g_pos.at(v, 0, j) = g;
        }
    }
    return g_pos;
}

dressi::Variable TransformToClipVar(const dressi::Variable& pos,
                                    const Mat4& mvp) {
    using namespace dressi;
    Variable x = F::GetX(pos);
    Variable y = F::GetY(pos);
    Variable z = F::GetZ(pos);
    const auto row = [&](int r) {
        return x * mvp[size_t(0 * 4 + r)] + y * mvp[size_t(1 * 4 + r)] +
               z * mvp[size_t(2 * 4 + r)] + mvp[size_t(3 * 4 + r)];
    };
    return F::Vec4(row(0), row(1), row(2), row(3));
}

void AdamStep(AdamState& state, std::vector<float>& params,
              const std::vector<float>& grad, float lr) {
    if (state.m.size() != params.size()) {
        state.m.assign(params.size(), 0.f);
        state.v.assign(params.size(), 0.f);
        state.t = 0;
    }
    state.t++;
    const float bc1 = 1.f - std::pow(state.beta1, float(state.t));
    const float bc2 = 1.f - std::pow(state.beta2, float(state.t));
    for (size_t i = 0; i < params.size(); i++) {
        state.m[i] = state.beta1 * state.m[i] + (1.f - state.beta1) * grad[i];
        state.v[i] = state.beta2 * state.v[i] +
                     (1.f - state.beta2) * grad[i] * grad[i];
        const float m_hat = state.m[i] / bc1;
        const float v_hat = state.v[i] / bc2;
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + state.eps);
    }
}

}  // namespace dressi_examples
