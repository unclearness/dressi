// One-shot Vulkan capability probe (own throwaway instance, RAII-cleaned)
// reported to Kotlin as JSON: gates silhouette on geometryShader and
// surfaces the limits that matter for Dressi on mobile GPUs.

#include <vkw/vkw.h>

#include <cstdio>
#include <string>

namespace dressi_android {

std::string QueryDeviceCapsJson() {
    try {
        auto instance = vkw::CreateInstance("dressi-caps", 1, "dressi-caps",
                                            1, /*debug_enable=*/false,
                                            /*surface_enable=*/false);
        const vk::PhysicalDevice phys = vkw::GetFirstPhysicalDevice(instance);
        const auto props = vkw::GetPhysicalProperties(phys);
        const auto feats = vkw::GetPhysicalFeatures2(phys);
        const vk::PhysicalDeviceLimits& lim = props->limits;

        char buf[1024];
        std::snprintf(
                buf, sizeof(buf),
                "{\"deviceName\":\"%s\","
                "\"apiVersion\":\"%u.%u.%u\","
                "\"driverVersion\":%u,"
                "\"vendorID\":%u,"
                "\"geometryShader\":%s,"
                "\"maxImageDimension2D\":%u,"
                "\"maxColorAttachments\":%u,"
                "\"maxPerStageDescriptorInputAttachments\":%u,"
                "\"maxPerStageDescriptorSampledImages\":%u,"
                "\"maxPerStageDescriptorStorageImages\":%u,"
                "\"maxComputeWorkGroupInvocations\":%u,"
                "\"timestampComputeAndGraphics\":%s}",
                props->deviceName.data(),
                VK_API_VERSION_MAJOR(props->apiVersion),
                VK_API_VERSION_MINOR(props->apiVersion),
                VK_API_VERSION_PATCH(props->apiVersion),
                props->driverVersion, props->vendorID,
                feats->features.geometryShader ? "true" : "false",
                lim.maxImageDimension2D, lim.maxColorAttachments,
                lim.maxPerStageDescriptorInputAttachments,
                lim.maxPerStageDescriptorSampledImages,
                lim.maxPerStageDescriptorStorageImages,
                lim.maxComputeWorkGroupInvocations,
                lim.timestampComputeAndGraphics ? "true" : "false");
        return buf;
    } catch (const std::exception& e) {
        return std::string("{\"error\":\"") + e.what() + "\"}";
    }
}

}  // namespace dressi_android
