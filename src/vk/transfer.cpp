#include "vk/transfer.h"

#include "codegen/glsl_codegen.h"

namespace dressi {

namespace {

// Grow-on-demand persistent staging. `cached` requests HostCached memory
// (falling back to plain coherent where unavailable) -- required for
// readback, where the CPU-side memcpy reads the mapped memory.
vkw::BufferPackPtr& EnsureStaging(const VkContext& ctx,
                                  vkw::BufferPackPtr& slot, size_t bytes,
                                  vk::BufferUsageFlags usage, bool cached) {
    if (slot && slot->size >= bytes) {
        return slot;
    }
    // Mutating the shared_ptr-held context's scratch: transfers are
    // synchronous and externally serialized (one queue)
    if (cached) {
        constexpr auto kCachedProps = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent |
                                      vk::MemoryPropertyFlagBits::eHostCached;
        try {
            slot = vkw::CreateBufferPack(ctx.physical_device, ctx.device,
                                         bytes, usage, kCachedProps);
            return slot;
        } catch (const std::runtime_error&) {
            // No cached+coherent memory type on this device; fall through
        }
    }
    slot = vkw::CreateBufferPack(ctx.physical_device, ctx.device, bytes,
                                 usage, vkw::HOST_VISIB_COHER_PROPS);
    return slot;
}

}  // namespace

vk::Format FormatOf(VType vtype) {
    switch (PhysChannels(vtype)) {
        case 1: return vk::Format::eR32Sfloat;
        case 2: return vk::Format::eR32G32Sfloat;
        case 4: return vk::Format::eR32G32B32A32Sfloat;
    }
    DRESSI_CHECK(false, "Unsupported VType for GPU image");
}

void SendImageToDevice(const VkContext& ctx, const vkw::ImagePackPtr& img,
                       const CpuImageView& cpu_img, VType vtype,
                       bool image_initialized) {
    const uint32_t n_logical = NumComponents(vtype);
    const uint32_t n_phys = PhysChannels(vtype);
    DRESSI_CHECK(cpu_img.channels == n_logical,
                 "sendImg: channel count does not match the Variable's VType");
    DRESSI_CHECK(cpu_img.width == img->view_size.width &&
                         cpu_img.height == img->view_size.height,
                 "sendImg: image size mismatch");

    const size_t n_pixels = size_t(cpu_img.width) * cpu_img.height;
    const size_t n_bytes = n_pixels * n_phys * sizeof(float);
    const auto& staging =
            EnsureStaging(ctx, ctx.send_staging, n_bytes,
                          vk::BufferUsageFlagBits::eTransferSrc,
                          /*cached=*/false);
    if (n_logical == n_phys) {
        vkw::SendToDevice(ctx.device, staging, cpu_img.data, n_bytes);
    } else {
        std::vector<float> phys(n_pixels * n_phys, 0.f);
        for (size_t p = 0; p < n_pixels; p++) {
            for (uint32_t c = 0; c < n_logical; c++) {
                phys[p * n_phys + c] = cpu_img.data[p * n_logical + c];
            }
        }
        vkw::SendToDevice(ctx.device, staging, phys.data(), n_bytes);
    }

    const auto init_layout = image_initialized
                                     ? vk::ImageLayout::eShaderReadOnlyOptimal
                                     : vk::ImageLayout::eUndefined;
    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        vkw::CopyBufferToImage(cmd, staging, img, init_layout,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    });
}

void SendGeometryToBuffer(const VkContext& ctx,
                          const vkw::BufferPackPtr& buf,
                          const CpuImageView& cpu_img, VType vtype) {
    DRESSI_CHECK(cpu_img.channels == NumComponents(vtype),
                 "sendImg: channel count does not match the Variable's VType");
    const size_t n_elems = cpu_img.numElems();
    DRESSI_CHECK(buf->size >= n_elems * 4, "Geometry buffer size mismatch");
    if (IsIntVType(vtype)) {
        std::vector<uint32_t> ints(n_elems);
        for (size_t i = 0; i < n_elems; i++) {
            ints[i] = uint32_t(int64_t(cpu_img.data[i]));
        }
        vkw::SendToDevice(ctx.device, buf, ints.data(), n_elems * 4);
    } else {
        vkw::SendToDevice(ctx.device, buf, cpu_img.data, n_elems * 4);
    }
}

CpuImage ReceiveImageFromDevice(const VkContext& ctx,
                                const vkw::ImagePackPtr& img, VType vtype) {
    const uint32_t n_logical = NumComponents(vtype);
    const uint32_t n_phys = PhysChannels(vtype);
    const uint32_t w = img->view_size.width;
    const uint32_t h = img->view_size.height;
    const size_t n_pixels = size_t(w) * h;

    const size_t n_bytes = n_pixels * n_phys * sizeof(float);
    const auto& staging =
            EnsureStaging(ctx, ctx.recv_staging, n_bytes,
                          vk::BufferUsageFlagBits::eTransferDst,
                          /*cached=*/true);
    RunOneShot(ctx, [&](const vk::UniqueCommandBuffer& cmd) {
        vkw::CopyImageToBuffer(cmd, img, staging,
                               vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    CpuImage out(w, h, n_logical);
    if (n_logical == n_phys) {
        vkw::RecvFromDevice(ctx.device, staging, out.data.data(), n_bytes);
    } else {
        std::vector<float> phys(n_pixels * n_phys);
        vkw::RecvFromDevice(ctx.device, staging, phys.data(), n_bytes);
        for (size_t p = 0; p < n_pixels; p++) {
            for (uint32_t c = 0; c < n_logical; c++) {
                out.data[p * n_logical + c] = phys[p * n_phys + c];
            }
        }
    }
    return out;
}

}  // namespace dressi
