#ifndef DRESSI_EXAMPLES_PBR_SHADING_RUN_H
#define DRESSI_EXAMPLES_PBR_SHADING_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Options: --mesh= (or positional), --env=, --out-dir= (default pbr_out),
// --size= (default 512), --frames= (0 = until the viewer closes;
// N = deterministic full orbit over N frames, forced to 60 when headless).
int RunPbrShading(const std::vector<std::string>& args, ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_PBR_SHADING_RUN_H
