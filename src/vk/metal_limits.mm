// Metal limits that Vulkan does not report, from the device's GPU family
// (Apple's Metal Feature Set Tables).
#import <Metal/Metal.h>

#include <string>

#include "vk/context.h"

namespace dressi {

MetalLimits QueryMetalLimits(const std::string& device_name) {
    MetalLimits limits;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
#if !TARGET_OS_IPHONE
        // A Mac can have several GPUs; use the one Vulkan picked.
        for (id<MTLDevice> d in MTLCopyAllDevices()) {
            if (device_name == d.name.UTF8String) {
                device = d;
            }
        }
#endif
        if (!device) {
            return limits;
        }
        // Tile memory holds a pass's color attachments on Apple GPUs. Mac2
        // (immediate-mode) GPUs have no such budget. The iOS simulator
        // presents itself as Apple2.
        if ([device supportsFamily:MTLGPUFamilyApple4]) {
            limits.max_color_bytes_per_pixel = 64;
        } else if ([device supportsFamily:MTLGPUFamilyApple2]) {
            limits.max_color_bytes_per_pixel = 32;
        }
        // MoltenVK reports the host GPU's limit under the simulator, where
        // textures are in fact capped lower.
        if (![device supportsFamily:MTLGPUFamilyApple3] &&
            [device supportsFamily:MTLGPUFamilyApple2]) {
            limits.max_image_dimension_2d = 8192;
        }
    }
    return limits;
}

}  // namespace dressi
