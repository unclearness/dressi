#ifndef DRESSI_EXAMPLES_IMAGE_FITTING_RUN_H
#define DRESSI_EXAMPLES_IMAGE_FITTING_RUN_H

#include "../common/example_host.h"

namespace dressi_examples {

// No options (the 64x64 fit runs in seconds everywhere); args are ignored.
int RunImageFitting(const std::vector<std::string>& args, ExampleHost& host);

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_IMAGE_FITTING_RUN_H
