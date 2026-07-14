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

// Unwelded unit-radius lat-long UV sphere with a proper equirectangular UV
// unwrap: uv = (phi/2pi, theta/pi). The seam (phi = 0 vs 2pi) and the poles
// carry DUPLICATED vertices (same position, distinct uv) so a texture does
// not smear across the seam; degenerate pole triangles are omitted. Because
// the seam/pole duplicates are separate vertices, this mesh is meant for
// texturing, NOT direct shape optimization (they would tear apart) -- weld
// it first with WeldByPosition for the geometry phase.
Mesh GenerateUvSphere(uint32_t stacks, uint32_t slices);

// Transfer a deformation from a coarse control mesh onto a finer mesh by
// direction. Both meshes must be star-shaped about the origin in their REST
// pose (spheres are). For each dst vertex, the ray from the origin through its
// position is intersected with the control mesh's rest triangles; the
// barycentric weights of that hit are applied to the control mesh's DEFORMED
// vertex positions. Lets a high-quality icosphere carry the shape while a
// lat-long UV sphere carries the parameterization -- and, being a function of
// direction alone, it maps the UV sphere's seam/pole DUPLICATES (which share a
// direction) to identical positions, so the seam stays closed with no welding.
dressi::CpuImage TransferDeformationByDirection(
        const dressi::CpuImage& ctrl_rest_pos,
        const dressi::CpuImage& ctrl_faces,
        const dressi::CpuImage& ctrl_deformed_pos,
        const dressi::CpuImage& dst_pos);

// Writes positions/faces as a Wavefront OBJ (uv omitted)
void SaveObjMesh(const std::string& path, const Mesh& mesh);

// Writes positions/uv/faces as a Wavefront OBJ (v/vt + f v/vt) plus a
// companion .mtl referencing `texture_filename` via map_Kd, so the textured
// result opens with its texture in MeshLab etc. The OBJ V coordinate is
// written flipped (1-v): dressi uv is top-left origin (image row 0), OBJ/GL
// samplers put v=0 at the bottom. `texture_filename` is stored verbatim in
// map_Kd, so keep it relative to the OBJ (e.g. "recovered_texture.png").
void SaveTexturedObjMesh(const std::string& path, const Mesh& mesh,
                         const std::string& texture_filename);

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
