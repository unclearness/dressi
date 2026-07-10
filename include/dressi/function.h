#ifndef DRESSI_FUNCTION_H
#define DRESSI_FUNCTION_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "dressi/types.h"
#include "dressi/variable.h"

namespace dressi {

struct FunctionNode;

// Backward generator: returns the gradient graph for input `xs[bwd_idx]`
// given output `y` and its incoming gradient `gy`. A null Variable means
// "no gradient" (non-backwardable path).
using BwdFunc = std::function<Variable(const Variables& xs, const Variable& y,
                                       const Variable& gy, uint32_t bwd_idx)>;

// Function object: an operation of the computational graph. Holds the GLSL
// snippet of its forward process and a generator for the backward pass.
class Function {
public:
    Function() = default;  // null handle
    Function(std::nullptr_t) {}
    Function(const std::string& code, ShaderType type, const BwdFunc& bwd_func);

    explicit operator bool() const { return m_node != nullptr; }
    bool operator==(const Function& o) const { return m_node == o.m_node; }
    bool operator!=(const Function& o) const { return m_node != o.m_node; }
    uint64_t id() const;

    // Getters
    std::string getFwdCode() const;
    ShaderType getShaderType() const;
    Variables getInputVars() const;
    Variable getOutputVar() const;

    // Building forward/backward connections
    Variable buildFwd(const Variables& xs);
    Variable buildBwd(const Variable& gy, uint32_t bwd_idx) const;

private:
    friend struct NodeAccess;
    explicit Function(std::shared_ptr<FunctionNode> node)
        : m_node(std::move(node)) {}
    std::shared_ptr<FunctionNode> m_node;
};

using Functions = std::vector<Function>;

}  // namespace dressi

template <>
struct std::hash<dressi::Function> {
    size_t operator()(const dressi::Function& f) const {
        return f ? std::hash<uint64_t>()(f.id()) : 0;
    }
};

template <>
struct std::less<dressi::Function> {
    bool operator()(const dressi::Function& a,
                    const dressi::Function& b) const {
        const uint64_t l = a ? a.id() : 0;
        const uint64_t r = b ? b.id() : 0;
        return l < r;
    }
};

#endif  // DRESSI_FUNCTION_H
