#ifndef DRESSI_EXAMPLES_SILHOUETTE_OPTIMIZATION_RUN_H
#define DRESSI_EXAMPLES_SILHOUETTE_OPTIMIZATION_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Options: --technique=hardsoftras|aa, --data-dir= (or positional),
// --out-dir=, --mesh=, --init-scale=, --views=, --screen=, --iters=,
// --sphere-level=, --samples=, --peels=, --single-view=, --no-view=,
// --lr=, --laplacian=, --normal= (see Options in run.cpp for defaults).
// Needs the geometryShader device feature (F::RasterizeFaceId).
int RunSilhouetteOptimization(const std::vector<std::string>& args,
                              ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_SILHOUETTE_OPTIMIZATION_RUN_H
