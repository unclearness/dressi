#ifndef DRESSI_EXAMPLES_EXAMPLE_HOST_H
#define DRESSI_EXAMPLES_EXAMPLE_HOST_H

#include <memory>
#include <string>
#include <vector>

#include "viewer.h"

namespace dressi_examples {

// Host services an example runs against. Desktop mains construct a
// DesktopHost; the Android app provides a host whose viewers blit to the
// app surface and whose cancelled() reflects the UI's Stop button (examples
// poll it once per iteration and exit through their normal teardown).
class ExampleHost {
public:
    virtual ~ExampleHost() = default;

    virtual std::unique_ptr<IViewer> makeViewer(uint32_t width,
                                                uint32_t height,
                                                const std::string& title) = 0;

    virtual bool cancelled() const = 0;
};

// Every example's entry point: argv-style options (without the program
// name) + the host. Returns the process exit code (0 ok, 1 quality gate
// failed, 2 exception).
using RunFn = int (*)(const std::vector<std::string>& args,
                      ExampleHost& host);

struct ExampleInfo {
    const char* name;
    RunFn run;
    // True for examples using F::RasterizeFaceId (gl_PrimitiveID needs the
    // geometryShader device feature, absent on some mobile GPUs).
    bool needs_geometry_shader;
};

const std::vector<ExampleInfo>& GetExampleRegistry();

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_EXAMPLE_HOST_H
