#ifndef DRESSI_EXAMPLES_GEOMETRY_UTILS_H
#define DRESSI_EXAMPLES_GEOMETRY_UTILS_H

#include <string>
#include <vector>

#include "asset_utils.h"
#include "dressi/dressi.h"
#include "dressi/mesh_utils.h"

namespace dressi_examples {

// Static-topology builders now live in the library (dressi/mesh_utils.h)
using dressi::BuildFaceAdjacency;
using dressi::BuildSoftGeometry;
using dressi::BuildVertexAdjacency;
using dressi::FaceNeighborsTex;
using dressi::SoftGeometry;
using dressi::VertexFacesTex;
using dressi::VertexNeighborsTex;

// Welded unit-radius icosphere (uv left zero): level 0 = icosahedron
// (12 v / 20 f), each level quadruples faces (level 3 = 642 v / 1280 f).
Mesh GenerateIcosphere(uint32_t level);

// Writes positions/faces as a Wavefront OBJ (uv omitted)
void SaveObjMesh(const std::string& path, const Mesh& mesh);

// Uniform-Laplacian regularization gradient {V,1,3}:
// lambda * (pos - mean(neighbors)); isolated vertices get zero
dressi::CpuImage UniformLaplacianGrad(
        const dressi::CpuImage& pos,
        const std::vector<std::vector<uint32_t>>& adj, float lambda);

// Normal-consistency regularization gradient {V,1,3} of
// lambda * sum_{adjacent faces (f,g)} (1 - n_f . n_g)
// (analytic through the normalized cross products; penalizes creases and
// flipped faces that a positional Laplacian cannot see)
dressi::CpuImage NormalConsistencyGrad(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj, float lambda);

// Number of adjacent face pairs whose normals disagree (n_f . n_g < 0) --
// a direct count of flipped/creased faces for diagnostics
uint32_t CountFlippedFacePairs(
        const dressi::CpuImage& pos, const dressi::CpuImage& faces,
        const std::vector<std::array<uint32_t, 2>>& face_adj);

// Chain rule through the (fixed) projection: g_pos {V,1,3} from the
// clip-position gradient {V,1,4} and the column-major MVP
dressi::CpuImage ChainRuleClipToPos(const dressi::CpuImage& g_clip,
                                    const Mat4& mvp);

// In-graph vertex transform: clip {V,1} VEC4 = mvp * (pos, 1) built from
// elementwise F:: ops (fusable; differentiable back to pos)
dressi::Variable TransformToClipVar(const dressi::Variable& pos,
                                    const Mat4& mvp);

// Adam optimizer state over a flat float parameter vector
struct AdamState {
    std::vector<float> m, v;
    int t = 0;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
};
void AdamStep(AdamState& state, std::vector<float>& params,
              const std::vector<float>& grad, float lr);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_GEOMETRY_UTILS_H
