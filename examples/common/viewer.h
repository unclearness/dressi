#ifndef DRESSI_EXAMPLES_VIEWER_H
#define DRESSI_EXAMPLES_VIEWER_H

#include <string>

#include "dressi/types.h"

namespace dressi_examples {

// Live-view abstraction: desktop = one GLFW window per instance (VkViewer),
// Android = one selectable stream blitted to the app's single Surface.
// Instances come from ExampleHost::makeViewer and are never null; a viewer
// that could not attach to a display reports valid() == false.
class IViewer {
public:
    virtual ~IViewer() = default;

    // False if no display is available (headless desktop, no app surface)
    virtual bool valid() const = 0;

    // Shows the image (1 or 3 channels, float [0,1]); returns false once
    // the view has been closed by the user.
    virtual bool update(const dressi::CpuImage& img) = 0;

    virtual void setTitle(const std::string& title) = 0;

    // Desktop: moves the window for deterministic layout. Android: no-op.
    virtual void setPosition(int x, int y) = 0;
};

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_VIEWER_H
