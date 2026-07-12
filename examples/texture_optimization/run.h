#ifndef DRESSI_EXAMPLES_TEXTURE_OPTIMIZATION_RUN_H
#define DRESSI_EXAMPLES_TEXTURE_OPTIMIZATION_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Options: --data-dir=<dir> (or positional; default data/bunny)
//          --out-dir=<dir> (default texopt_out)
//          --views=<n> (default 6), --iters=<n> (default 4000)
int RunTextureOptimization(const std::vector<std::string>& args,
                           ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_TEXTURE_OPTIMIZATION_RUN_H
