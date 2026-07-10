// Phase-0 smoke test: headless Vulkan bring-up through vkw.
// Creates an instance/device without any window or swapchain, clears an
// RGBA32F image on the GPU, reads it back, and checks pixel values.
// Also verifies that the formats Dressi-AD relies on support the usages
// required by the executor design.

#include <gtest/gtest.h>
#include <vkw/vkw.h>

#include <vector>

namespace {

struct VkCtx {
    vkw::InstancePackPtr instance;
    vk::PhysicalDevice physical_device;
    uint32_t queue_family_idx = 0;
    vk::UniqueDevice device;
    vk::Queue queue;
};

VkCtx CreateHeadlessContext() {
    VkCtx ctx;
    ctx.instance = vkw::CreateInstance("dressi_smoke", 1, "dressi", 1,
                                       /*debug_enable=*/false,
                                       /*surface_enable=*/false);
    ctx.physical_device = vkw::GetFirstPhysicalDevice(ctx.instance);
    ctx.queue_family_idx = vkw::GetQueueFamilyIdx(ctx.physical_device);
    ctx.device = vkw::CreateDevice(ctx.queue_family_idx, ctx.physical_device, 1,
                                   /*swapchain_support=*/false);
    ctx.queue = vkw::GetQueue(ctx.device, ctx.queue_family_idx);
    return ctx;
}

}  // namespace

TEST(VkSmoke, HeadlessDeviceCreation) {
    VkCtx ctx = CreateHeadlessContext();
    ASSERT_TRUE(ctx.instance);
    ASSERT_TRUE(static_cast<bool>(ctx.device));
    auto props = vkw::GetPhysicalProperties(ctx.physical_device);
    ASSERT_TRUE(props);
    EXPECT_GT(props->limits.maxColorAttachments, 0u);
}

TEST(VkSmoke, RequiredFormatsSupported) {
    VkCtx ctx = CreateHeadlessContext();
    const vk::FormatFeatureFlags required =
            vk::FormatFeatureFlagBits::eColorAttachment |
            vk::FormatFeatureFlagBits::eSampledImage |
            vk::FormatFeatureFlagBits::eTransferSrc |
            vk::FormatFeatureFlagBits::eTransferDst;
    for (const auto format :
         {vk::Format::eR32Sfloat, vk::Format::eR32G32Sfloat,
          vk::Format::eR32G32B32A32Sfloat, vk::Format::eR32Sint,
          vk::Format::eR32G32Sint, vk::Format::eR32G32B32A32Sint}) {
        const auto props = ctx.physical_device.getFormatProperties(format);
        EXPECT_EQ(props.optimalTilingFeatures & required, required)
                << "format: " << vk::to_string(format);
    }
}

TEST(VkSmoke, ClearImageAndReadback) {
    VkCtx ctx = CreateHeadlessContext();

    const vk::Extent2D size = {16, 8};
    const uint32_t n_pixels = size.width * size.height;
    auto img = vkw::CreateImagePack(
            ctx.physical_device, ctx.device, vk::Format::eR32G32B32A32Sfloat,
            size, 1,
            vk::ImageUsageFlagBits::eColorAttachment |
                    vk::ImageUsageFlagBits::eSampled |
                    vk::ImageUsageFlagBits::eTransferSrc |
                    vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
    ASSERT_TRUE(img);

    auto buf = vkw::CreateBufferPack(ctx.physical_device, ctx.device,
                                     n_pixels * 4 * sizeof(float),
                                     vk::BufferUsageFlagBits::eTransferDst,
                                     vkw::HOST_VISIB_COHER_PROPS);
    ASSERT_TRUE(buf);

    auto cmd_pack =
            vkw::CreateCommandBuffersPack(ctx.device, ctx.queue_family_idx, 1);
    const auto& cmd = cmd_pack->cmd_bufs[0];

    vkw::BeginCommand(cmd, /*one_time_submit=*/true);
    const std::array<float, 4> clear_col = {0.25f, 0.5f, 0.75f, 1.f};
    vkw::ClearColorImage(cmd, img, vk::ClearColorValue(clear_col),
                         vk::ImageLayout::eUndefined,
                         vk::ImageLayout::eTransferSrcOptimal);
    vkw::CopyImageToBuffer(cmd, img, buf,
                           vk::ImageLayout::eTransferSrcOptimal);
    vkw::EndCommand(cmd);

    auto fence = vkw::CreateFence(ctx.device);
    vkw::QueueSubmit(ctx.queue, cmd, fence);
    ASSERT_EQ(vkw::WaitForFence(ctx.device, fence), vk::Result::eSuccess);

    std::vector<float> result(n_pixels * 4, 0.f);
    vkw::RecvFromDevice(ctx.device, buf, result.data(),
                        result.size() * sizeof(float));

    for (uint32_t i = 0; i < n_pixels; i++) {
        EXPECT_FLOAT_EQ(result[i * 4 + 0], 0.25f) << "pixel " << i;
        EXPECT_FLOAT_EQ(result[i * 4 + 1], 0.5f) << "pixel " << i;
        EXPECT_FLOAT_EQ(result[i * 4 + 2], 0.75f) << "pixel " << i;
        EXPECT_FLOAT_EQ(result[i * 4 + 3], 1.f) << "pixel " << i;
    }
}
