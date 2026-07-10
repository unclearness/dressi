#ifndef DRESSI_MESH_UTILS_H
#define DRESSI_MESH_UTILS_H

#include <array>
#include <vector>

#include "dressi/types.h"

namespace dressi {

// CPU-side static-topology builders. Meshes enter the graph as CpuImage
// leaves ({V,1} / {F,1} arrays, indices stored as floats); these helpers
// derive the adjacency textures and unwelded geometry that the raster /
// gather ops consume. All are upload-once data for a fixed topology.

// Vertex -> neighbor-vertex adjacency from the face list ({F,1,3} float ids)
std::vector<std::vector<uint32_t>> BuildVertexAdjacency(const CpuImage& faces,
                                                        uint32_t n_verts);

// Adjacent face pairs sharing an edge (each unordered pair once)
std::vector<std::array<uint32_t, 2>> BuildFaceAdjacency(const CpuImage& faces);

// Vertex -> incident-face adjacency as a {V, max_degree} 1-channel image
// (face index as float, -1 padding). Static topology; upload once and the
// gather backwards skip their O(V*F) face scan.
CpuImage VertexFacesTex(const CpuImage& faces, uint32_t n_verts);

// Vertex -> neighbor-vertex adjacency as a {V, max_degree} 1-channel image
// (vertex index as float, -1 padding); feeds F::VertexNeighborMean
CpuImage VertexNeighborsTex(const CpuImage& faces, uint32_t n_verts);

// Face -> edge-adjacent-face ids as a VEC3-channel {F,1} image (-1 for
// boundary edges); feeds F::NormalConsistencyFaceTerm
CpuImage FaceNeighborsTex(const CpuImage& faces);

// Per-face unwelded soft geometry for HardSoftRas: triangles enlarged by
// `radius_px` in screen space (scaled about the centroid), face index as a
// per-vertex attribute, and sequential faces (3i, 3i+1, 3i+2).
// Behind-camera faces (w <= eps) are passed through un-enlarged.
struct SoftGeometry {
    CpuImage clip;     // {3F,1,4}
    CpuImage face_id;  // {3F,1,1}
    CpuImage faces;    // {F,1,3}
};
SoftGeometry BuildSoftGeometry(const CpuImage& hard_clip,
                               const CpuImage& faces, ImgSize screen,
                               float radius_px);

// Per-vertex inverse-UV inputs (for F::Texture's gradient gather):
//   UvAsClip:   {V,1,4} = (u*2-1, v*2-1, 0.5, 1)   (rasterize in UV space)
//   ScreenAttr: {V,1,4} = (screen_x, screen_y, 1, 0) from clip positions
CpuImage UvAsClip(const CpuImage& uv);
CpuImage ScreenAttr(const CpuImage& clip, ImgSize screen);

}  // namespace dressi

#endif  // DRESSI_MESH_UTILS_H
