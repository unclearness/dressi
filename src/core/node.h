#ifndef DRESSI_CORE_NODE_H
#define DRESSI_CORE_NODE_H

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "dressi/function.h"
#include "dressi/types.h"
#include "dressi/variable.h"

namespace dressi {

// How a function reads each of its inputs (drives tex/slt/inp classification
// during substage packing)
enum class InputAccess {
    SamePixel,   // reads the same pixel coordinate -> fusable / input attachment
    Sampled,     // texture(sampler, uv) -> texture sampler input
    TexelFetch,  // texelFetch at arbitrary coords -> sampler-less texture input
};

struct CpuTensor {
    VType vtype = FLOAT;
    ImgSize size = {1, 1};
    // components-interleaved, row-major: size.w * size.h * NumComponents(vtype)
    std::vector<float> data;

    uint32_t numComp() const { return NumComponents(vtype); }
    size_t numPixels() const { return size_t(size.w) * size.h; }
};

// Whole-tensor CPU kernel: the test oracle and constant folding backend
using CpuFunc =
        std::function<CpuTensor(const std::vector<CpuTensor>& xs)>;

// Custom output type/size inference; nullptr selects the default
// (component-wise broadcasting) rules
using InferFn = std::function<std::pair<VType, ImgSize>(const Variables& xs)>;

struct FunctionNode {
    uint64_t id = 0;
    std::string name;  // op name for debugging and golden tests
    std::string fwd_code;
    ShaderType shader_type = FRAG;
    BwdFunc bwd_func;
    CpuFunc cpu_func;
    InferFn infer;  // nullptr -> default component-wise rules
    std::vector<InputAccess> input_access;  // empty -> all SamePixel
    std::vector<std::shared_ptr<VariableNode>> xs;  // strong: inputs are older
    std::weak_ptr<VariableNode> y;                  // weak: forward edge
};

struct VariableNode {
    uint64_t id = 0;
    VType vtype = FLOAT;
    ImgSize img_size = {1, 1};
    std::shared_ptr<FunctionNode> creator;           // strong: creator is older
    std::vector<std::weak_ptr<FunctionNode>> users;  // weak: forward edges
    bool req_grad = false;
    bool is_dirty = true;
    // Set for constant ops (F::Float etc.); enables constant folding
    std::optional<std::vector<float>> const_val;
};

uint64_t NextNodeId();

// Internal bridge between public handles and shared nodes
struct NodeAccess {
    static Variable Wrap(std::shared_ptr<VariableNode> node) {
        return Variable(std::move(node));
    }
    static Function Wrap(std::shared_ptr<FunctionNode> node) {
        return Function(std::move(node));
    }
    static const std::shared_ptr<VariableNode>& Node(const Variable& v) {
        return v.m_node;
    }
    static const std::shared_ptr<FunctionNode>& Node(const Function& f) {
        return f.m_node;
    }
};

}  // namespace dressi

#endif  // DRESSI_CORE_NODE_H
