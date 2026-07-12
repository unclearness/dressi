#ifndef DRESSI_EXAMPLES_DESKTOP_HOST_H
#define DRESSI_EXAMPLES_DESKTOP_HOST_H

#include "example_host.h"

namespace dressi_examples {

// Desktop host: viewers are VkViewer GLFW windows; never cancelled.
class DesktopHost : public ExampleHost {
public:
    std::unique_ptr<IViewer> makeViewer(uint32_t width, uint32_t height,
                                        const std::string& title) override;
    bool cancelled() const override { return false; }
};

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_DESKTOP_HOST_H
