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

    // Persistent transfer staging, grown on demand (every transfer is a
    // synchronous one-shot submit, so one buffer per direction suffices).
    // The readback buffer prefers HostCached memory: memcpy FROM
    // write-combined HostCoherent memory runs at ~100 MB/s and dominated
    // recvImg before this cache existed.
    mutable vkw::BufferPackPtr send_staging;
    mutable vkw::BufferPackPtr recv_staging;
    // Reused one-shot command buffer + fence (RunOneShot used to create a
    // command pool per transfer)
    mutable vkw::CommandBuffersPackPtr oneshot_cmds;
    mutable vkw::FencePtr oneshot_fence;
    // Scratch command buffers for the combined upload+exec+download submit
    mutable vkw::CommandBuffersPackPtr exec_transfer_cmds;
};
using VkContextPtr = std::shared_ptr<VkContext>;

VkContextPtr CreateVkContext(bool debug_enable = false);

// Runs `record` into a one-shot command buffer, submits, and waits.
void RunOneShot(const VkContext& ctx,
                const std::function<void(const vk::UniqueCommandBuffer&)>& record);

}  // namespace dressi

#endif  // DRESSI_VK_CONTEXT_H
