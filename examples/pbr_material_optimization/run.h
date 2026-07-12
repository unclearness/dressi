#ifndef DRESSI_EXAMPLES_PBR_MATERIAL_OPTIMIZATION_RUN_H
#define DRESSI_EXAMPLES_PBR_MATERIAL_OPTIMIZATION_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Options: --mesh=, --env=, --out-dir= (default pbrmat_out),
// --optimize=albedo|mr, --size= (256), --tex= (512), --views= (6),
// --iters= (2000), --lr= (0.02).
int RunPbrMaterialOptimization(const std::vector<std::string>& args,
                               ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_PBR_MATERIAL_OPTIMIZATION_RUN_H
