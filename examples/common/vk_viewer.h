#ifndef DRESSI_EXAMPLES_VK_VIEWER_H
#define DRESSI_EXAMPLES_VK_VIEWER_H

#include <memory>
#include <string>

#include "dressi/types.h"

namespace dressi_examples {

// Minimal live-view window built on glad's Vulkan loader + GLFW, fully
// separate from Dressi's Vulkan context: each update uploads a CpuImage
// through a staging buffer and copies it to the swapchain (nearest-scaled
// on the CPU to the window size). Intended as a debug viewer; keep its cost
// out of optimization timing by measuring around update() separately.
class VkViewer {
public:
    VkViewer(uint32_t width, uint32_t height, const std::string& title);
    ~VkViewer();
    VkViewer(const VkViewer&) = delete;
    VkViewer& operator=(const VkViewer&) = delete;

    // False if the window could not be created (e.g. headless environment)
    bool valid() const;

    // Shows the image (1 or 3 channels, float [0,1]); returns false once
    // the window has been closed by the user.
    bool update(const dressi::CpuImage& img);

    void setTitle(const std::string& title);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_VK_VIEWER_H
