// Silhouette optimization: deform an icosphere toward the bunny using
// vertex-position gradients. Two techniques (selected by --technique):
//   hardsoftras - HardSoftRas (Dressi paper, Alg.2): soft rasterization of
//                 enlarged triangles, sigmoid of the signed distance
//   aa          - screen-space AA (Dr.Hair, Eq.3-5): hard rasterization +
//                 differentiable edge blending over the triangle-ID buffer
// Placeholder: filled in once the RasterizeSoft / AntiAlias ops land.

#include <cstdio>

#include "../common/geometry_utils.h"

int main(int /*argc*/, char** /*argv*/) {
    const auto sphere = dressi_examples::GenerateIcosphere(3);
    std::printf("icosphere level 3: %u verts, %u faces\n",
                sphere.numVertices(), sphere.numFaces());
    std::printf("silhouette_optimization: not implemented yet\n");
    return 1;
}
