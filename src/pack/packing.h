#ifndef DRESSI_PACK_PACKING_H
#define DRESSI_PACK_PACKING_H

#include <cstdint>
#include <map>

#include "pack/substage.h"

namespace dressi {

// Vulkan resource limits constraining a single substage / stage. Values are
// clamped from VkPhysicalDeviceLimits at context creation and can be lowered
// artificially in tests to force degenerate packings.
struct PackingLimits {
    uint32_t max_input_attachments = 8;
    uint32_t max_output_attachments = 8;
    uint32_t max_sampled_images = 32;
    uint32_t max_stage_attachments = 16;  // per render pass, all kinds
};

// Greedy substage packing (Appendix A "Substage Packing"): packs `funcs`
// back-to-front with a greedy edge-count heuristic under `limits`.
// `root_vars` are variables demanded externally (loss, optimizer outputs):
// they always become substage outputs.
SubStages PackFunctionsIntoSubStages(const Functions& funcs,
                                     const Variables& root_vars,
                                     const PackingLimits& limits);

// Greedy stage packing (Appendix A "Stage Packing"): merges consecutive
// compatible substages into stages (render passes).
Stages PackSubStagesIntoStages(SubStages substages,
                               const PackingLimits& limits);

// Convenience: both packing steps
Stages GreedyPack(const Functions& funcs, const Variables& root_vars,
                  const PackingLimits& limits);

}  // namespace dressi

#endif  // DRESSI_PACK_PACKING_H
