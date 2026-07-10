#ifndef DRESSI_CORE_OP_BUILDER_H
#define DRESSI_CORE_OP_BUILDER_H

#include <optional>
#include <string>
#include <vector>

#include "core/node.h"

namespace dressi {

// Internal extended construction used by the F:: operators. Wraps the
// appendix-style Function(code, type, bwd).buildFwd(xs) flow while attaching
// the CPU kernel, custom inference, and input access annotations.
struct OpDesc {
    std::string name;
    std::string fwd_code;
    ShaderType shader_type = FRAG;
    BwdFunc bwd;
    CpuFunc cpu;
    InferFn infer;                          // nullptr -> elementwise default
    std::vector<InputAccess> input_access;  // empty -> all SamePixel
    std::optional<std::vector<float>> const_val;
};

Variable MakeOp(OpDesc desc, const Variables& xs);

// Per-pixel component kernel -> whole-tensor CpuFunc with {1,1} spatial and
// scalar->vector component broadcasting (mirrors GLSL semantics).
// `out_vtype`: nullopt selects the widest-input rule (elementwise default).
using CpuPixelKernel = std::function<void(
        const std::vector<const float*>& xs, const std::vector<uint32_t>& ns,
        float* y, uint32_t ny)>;
CpuFunc ElementwiseCpu(CpuPixelKernel kernel,
                       std::optional<VType> out_vtype = std::nullopt);

// Convenience: single-output-component-from-same-index kernels (pure map)
CpuFunc MapCpu1(float (*f)(float));
CpuFunc MapCpu2(float (*f)(float, float));
CpuFunc MapCpu3(float (*f)(float, float, float));

InferFn FixedInfer(VType vtype);  // fixed vtype, common img size

}  // namespace dressi

#endif  // DRESSI_CORE_OP_BUILDER_H
