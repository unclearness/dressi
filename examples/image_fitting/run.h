#ifndef DRESSI_EXAMPLES_IMAGE_FITTING_RUN_H
#define DRESSI_EXAMPLES_IMAGE_FITTING_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// Options: --out-dir= (default imgfit_out; receives only bench.json —
// the 64x64 fit itself produces no image outputs).
int RunImageFitting(const std::vector<std::string>& args, ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_IMAGE_FITTING_RUN_H
