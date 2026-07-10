#ifndef DRESSI_CORE_CPU_EVAL_H
#define DRESSI_CORE_CPU_EVAL_H

#include <map>

#include "core/node.h"

namespace dressi {

// CPU reference interpreter for computational graphs - the test oracle.
// Evaluates `target` by topologically walking creator functions and invoking
// their CpuFuncs. `leaf_bindings` provides values for variables without a
// creator (leaves); constant variables use their recorded const_val.
class CpuEvaluator {
public:
    void bind(const Variable& var, CpuTensor value);
    void bindImage(const Variable& var, const CpuImage& img);
    CpuTensor eval(const Variable& target);
    void clearCache();

private:
    std::map<Variable, CpuTensor> m_bindings;
    std::map<Variable, CpuTensor> m_cache;
};

CpuTensor CpuTensorFromImage(const CpuImage& img, VType vtype);
CpuImage CpuImageFromTensor(const CpuTensor& tensor);

}  // namespace dressi

#endif  // DRESSI_CORE_CPU_EVAL_H
