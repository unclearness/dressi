#ifndef DRESSI_DRESSI_AD_H
#define DRESSI_DRESSI_AD_H

#include <functional>
#include <memory>

#include "dressi/types.h"
#include "dressi/variable.h"

namespace dressi {

// DressiAD: DR-specialized AD library class (Appendix A of the Dressi paper).
// Owns the Vulkan context and the staged build pipeline of execStep():
// backward construction, optimizer wiring, graph traversal, substage/stage
// packing, Vulkan object creation, and command execution.
class DressiAD {
public:
    // Optimizer function
    // (takes inputs and their gradients, and returns optimized outputs)
    using Optimizer = std::function<Variables(Variables xs, Variables gxs)>;

    // Building status
    enum InitStatus {
        BACKWARD,
        OPTIMIZER,
        TRAVERSE,
        SUBSTAGE,
        STAGE,
        VULKAN,
        FINISHED,
    };

    enum class PackingMode {
        Naive,  // 1 function = 1 pass (correctness baseline)
        RSP,    // reactive shader packing
    };

    DressiAD();
    ~DressiAD();
    DressiAD(const DressiAD&) = delete;
    DressiAD& operator=(const DressiAD&) = delete;

    // Setters
    void setLossVar(const Variable& loss_var);
    void setOptimizer(const Optimizer& optim_func);
    void setPackingMode(PackingMode mode);
    // Reactive cache thresholds: after the dirty pattern has been stable for
    // `fast` iterations a fast rebuild (stage packing only) prunes cached
    // branches; after `full` iterations a full rebuild (substage packing)
    // does. Pass 0 to disable either.
    void setRebuildCounts(uint32_t fast, uint32_t full);

    // Forces the variable to stay a substage output so recvImg() can read
    // it even when packing would otherwise fuse it into a shader. Call
    // before the first execStep() (or a rebuild follows).
    void markOutput(const Variable& var);

    // Introspection (current build)
    size_t getStageCount() const;
    size_t getSubStageCount() const;

    // Execute one step of rendering and optimization
    void execStep();

    // Transfer image data between CPU and GPU. Sends before the first
    // execStep() are kept pending and flushed once images exist.
    void sendImg(const Variable& var, const CpuImage& cpu_img);
    CpuImage recvImg(const Variable& var);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace dressi

#endif  // DRESSI_DRESSI_AD_H
