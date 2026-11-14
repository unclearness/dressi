#ifndef DRESSI_EXAMPLES_EXAMPLE_HOST_H
#define DRESSI_EXAMPLES_EXAMPLE_HOST_H

#include <cstdint>
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
    // geometryShader device feature, absent on some mobile GPUs). MoltenVK
    // reports it absent yet gl_PrimitiveID works, so iOS ignores this.
    bool needs_geometry_shader;
};

const std::vector<ExampleInfo>& GetExampleRegistry();

// What --no-view=1 gets instead of a host viewer: nothing is created on the
// display side, so the run is the `none` benchmark condition.
class NullViewer final : public IViewer {
public:
    bool valid() const override { return false; }
    bool update(const dressi::CpuImage&) override { return false; }
    void setTitle(const std::string&) override {}
    void setPosition(int, int) override {}
};

inline std::unique_ptr<IViewer> MakeViewer(ExampleHost& host, bool no_view,
                                           uint32_t width, uint32_t height,
                                           const std::string& title) {
    if (no_view) {
        return std::make_unique<NullViewer>();
    }
    return host.makeViewer(width, height, title);
}

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_EXAMPLE_HOST_H
