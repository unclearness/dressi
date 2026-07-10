#ifndef DRESSI_VK_CONTEXT_H
#define DRESSI_VK_CONTEXT_H

#include <vkw/vkw.h>

#include <memory>

namespace dressi {

// Headless Vulkan bring-up shared by the executor and transfer layer.
struct VkContext {
    vkw::InstancePackPtr instance;
    vk::PhysicalDevice physical_device;
    uint32_t queue_family_idx = 0;
    vk::UniqueDevice device;
    vk::Queue queue;
    vk::PhysicalDeviceLimits limits;
    std::unique_ptr<vkw::GLSLCompiler> glsl_compiler;
};
using VkContextPtr = std::shared_ptr<VkContext>;

VkContextPtr CreateVkContext(bool debug_enable = false);

// Runs `record` into a one-shot command buffer, submits, and waits.
void RunOneShot(const VkContext& ctx,
                const std::function<void(const vk::UniqueCommandBuffer&)>& record);

}  // namespace dressi

#endif  // DRESSI_VK_CONTEXT_H
