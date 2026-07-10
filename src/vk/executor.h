#ifndef DRESSI_VK_EXECUTOR_H
#define DRESSI_VK_EXECUTOR_H

#include <vkw/vkw.h>

#include <map>

#include "dressi/variable.h"
#include "pack/substage.h"
#include "vk/context.h"

namespace dressi {

// Built Vulkan objects for a packed stage list ("ParseStagesAsVulkanObjects"
// of the Dressi paper). Images persist across rebuilds via the map passed in.
struct GpuPlan {
    std::map<Variable, vkw::ImagePackPtr> imgs;
    std::map<Variable, vkw::TexturePackPtr> textures;  // nearest samplers
    // Vertex/index buffers for RASTER inputs (leaf geometry data)
    std::map<Variable, vkw::BufferPackPtr> vtx_bufs;
    // One vec4 UBO per {1,1} uniform input (uif_vars), refreshed on the GPU
    // by an image->buffer copy after the producing stage
    std::map<Variable, vkw::BufferPackPtr> uif_bufs;
    std::vector<vkw::ImagePackPtr> depth_imgs;  // one per RASTER stage
    std::vector<vkw::RenderPassPackPtr> render_passes;
    std::vector<vkw::FrameBufferPackPtr> frame_buffers;
    std::vector<vkw::PipelinePackPtr> pipelines;
    std::vector<vkw::DescSetPackPtr> desc_sets;
    vkw::CommandBuffersPackPtr cmd_pack;
    vkw::FencePtr fence;
    // Per-stage GPU timing (created only when debug logging is enabled):
    // timestamps around every stage; averaged and reported periodically
    vk::UniqueQueryPool ts_pool;
    std::vector<std::string> ts_labels;   // one per stage + tail sections
    std::vector<double> ts_accum_us;      // running sums per interval
    uint32_t ts_frames = 0;
    float ts_period_ns = 0.f;
};

// Creates images and geometry buffers (reusing the previous plan's entries
// so GPU-resident data survives rebuilds), builds render passes/pipelines/
// descriptor sets from the stages, and records the full command buffer.
GpuPlan BuildGpuPlan(const VkContext& ctx, const Stages& stages,
                     const std::map<Variable, Variable>& upd_inp_map,
                     GpuPlan prev_plan);

// Submits the recorded command buffer and waits for completion.
void ExecuteGpuPlan(const VkContext& ctx, GpuPlan& plan);

// Records `uploads` into a scratch command buffer and submits it together
// with the plan's command buffer in ONE vkQueueSubmit + one fence wait --
// the eager binding path uploads changed leaves and executes every
// iteration, and a combined submit removes the separate upload fence
// wait. The uploads leave images in ShaderReadOnlyOptimal, which the plan
// reads (the layout-transition barrier orders them within the batch).
struct ImageSendItem;  // vk/transfer.h
void ExecuteGpuPlanWithUploads(const VkContext& ctx, GpuPlan& plan,
                               const std::vector<ImageSendItem>& uploads);

}  // namespace dressi

#endif  // DRESSI_VK_EXECUTOR_H
