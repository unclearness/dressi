#include "vk/context.h"

#include "dressi/dressi_ad.h"

#include <algorithm>
#include <cstdlib>

namespace dressi {

void PrepareVulkanLoaderEnv() {
#if defined(__APPLE__)
    // MoltenVK misreads shaders that mix input attachments (bound directly)
    // with descriptors in a Metal argument buffer: garbage 8x8 blocks on
    // macOS, wrong values on the iOS simulator. RSP fusion produces exactly
    // that mix. Read at instance creation, so this must come first.
    setenv("MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", "0", /*overwrite=*/0);
#endif
#if defined(DRESSI_MOLTENVK_ICD)  // set for macOS, not iOS (no loader)
    if (!std::getenv("VK_DRIVER_FILES") && !std::getenv("VK_ICD_FILENAMES")) {
        setenv("VK_DRIVER_FILES", DRESSI_MOLTENVK_ICD, /*overwrite=*/0);
    }
    setenv("VK_LAYER_PATH", DRESSI_VK_LAYER_DIR, /*overwrite=*/0);
#endif
}

VkContextPtr CreateVkContext(bool debug_enable) {
    PrepareVulkanLoaderEnv();
    auto ctx = std::make_shared<VkContext>();
    ctx->instance = vkw::CreateInstance("dressi", 1, "dressi", 1, debug_enable,
                                        /*surface_enable=*/false);
    ctx->physical_device = vkw::GetFirstPhysicalDevice(ctx->instance);
    // Do NOT require the explicit TRANSFER bit (vkw's default): per the
    // spec, graphics or compute capability implies transfer support, and
    // Adreno omits the bit on its graphics+compute family ("No sufficient
    // queue" on Snapdragon otherwise).
    ctx->queue_family_idx = vkw::GetQueueFamilyIdx(
            ctx->physical_device,
            vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
    // Enable the physical device's feature set (gl_PrimitiveID in fragment
    // shaders needs the Geometry capability from the geometryShader feature)
    ctx->device = vkw::CreateDevice(ctx->queue_family_idx,
                                    ctx->physical_device, 1,
                                    /*swapchain_support=*/false,
                                    vkw::GetPhysicalFeatures2(
                                            ctx->physical_device));
    ctx->queue = vkw::GetQueue(ctx->device, ctx->queue_family_idx);
    ctx->limits = vkw::GetPhysicalProperties(ctx->physical_device)->limits;
#if defined(__APPLE__)
    const auto props = vkw::GetPhysicalProperties(ctx->physical_device);
    const MetalLimits metal = QueryMetalLimits(props->deviceName.data());
    ctx->max_color_bytes_per_pixel = metal.max_color_bytes_per_pixel;
    ctx->limits.maxImageDimension2D = std::min(
            ctx->limits.maxImageDimension2D, metal.max_image_dimension_2d);
#endif
    ctx->glsl_compiler = std::make_unique<vkw::GLSLCompiler>();
    return ctx;
}

void RunOneShot(
        const VkContext& ctx,
        const std::function<void(const vk::UniqueCommandBuffer&)>& record) {
    if (!ctx.oneshot_cmds) {
        ctx.oneshot_cmds = vkw::CreateCommandBuffersPack(
                ctx.device, ctx.queue_family_idx, 1);
        ctx.oneshot_fence = vkw::CreateFence(ctx.device);
    }
    const auto& cmd = ctx.oneshot_cmds->cmd_bufs[0];
    vkw::ResetCommand(cmd);
    vkw::BeginCommand(cmd, /*one_time_submit=*/true);
    record(cmd);
    vkw::EndCommand(cmd);
    vkw::ResetFence(ctx.device, ctx.oneshot_fence);
    vkw::QueueSubmit(ctx.queue, cmd, ctx.oneshot_fence);
    vkw::WaitForFence(ctx.device, ctx.oneshot_fence);
}

}  // namespace dressi
