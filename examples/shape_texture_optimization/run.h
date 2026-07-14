#ifndef DRESSI_EXAMPLES_SHAPE_TEXTURE_OPTIMIZATION_RUN_H
#define DRESSI_EXAMPLES_SHAPE_TEXTURE_OPTIMIZATION_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Two-phase demo: fit a UV-unwrapped sphere to the bunny.
//   Phase 1 (shape):   multi-view screen-space AA silhouette optimization
//                      deforms a uniform icosphere toward the bunny
//                      silhouette (vertex-position gradients). The icosphere
//                      is a far better deformation cage than a lat-long UV
//                      sphere's stretched pole triangles.
//   Phase 2 (texture): the icosphere deformation is transferred onto the
//                      UV-carrying sphere by direction, then a texture atlas
//                      in the sphere's UV space is optimized so the textured
//                      sphere matches renders of the textured bunny.
//
// Options: --data-dir=<dir> (or positional; default data/bunny)
//          --out-dir=<dir>  (default shapetex_out)
//          --views=<n>      (default 8)
//          --shape-iters=<n>   (default 200)
//          --texture-iters=<n> (default 300)
//          --sphere-level=<n>  (icosphere subdivision for shape; default 4)
//          --stacks=<n> --slices=<n> (UV-sphere resolution; default 32/64)
//          --view-interval=<n> (live-viewer cadence; 0 = off)
//          --no-view
int RunShapeTextureOptimization(const std::vector<std::string>& args,
                                ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_SHAPE_TEXTURE_OPTIMIZATION_RUN_H
