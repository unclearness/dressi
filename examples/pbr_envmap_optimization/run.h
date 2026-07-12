#ifndef DRESSI_EXAMPLES_PBR_ENVMAP_OPTIMIZATION_RUN_H
#define DRESSI_EXAMPLES_PBR_ENVMAP_OPTIMIZATION_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Options: --mesh=, --env=, --out-dir= (default pbrenv_out),
// --size= (192), --envres= (64), --views= (4), --iters= (1500), --lr= (0.05).
int RunPbrEnvmapOptimization(const std::vector<std::string>& args,
                             ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_PBR_ENVMAP_OPTIMIZATION_RUN_H
