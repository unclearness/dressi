#include "core/cpu_eval.h"

#include <functional>

namespace dressi {

void CpuEvaluator::bind(const Variable& var, CpuTensor value) {
    DRESSI_CHECK(var, "Null Variable");
    DRESSI_CHECK(value.vtype == var.getVType(), "bind: vtype mismatch");
    DRESSI_CHECK(value.size == var.getImgSize(), "bind: image size mismatch");
    DRESSI_CHECK(value.data.size() ==
                         value.numPixels() * value.numComp(),
                 "bind: data size mismatch");
    m_bindings[var] = std::move(value);
}

void CpuEvaluator::bindImage(const Variable& var, const CpuImage& img) {
    bind(var, CpuTensorFromImage(img, var.getVType()));
}

void CpuEvaluator::clearCache() {
    m_cache.clear();
}

CpuTensor CpuEvaluator::eval(const Variable& target) {
    DRESSI_CHECK(target, "Null Variable");
    if (auto it = m_bindings.find(target); it != m_bindings.end()) {
        return it->second;
    }
    if (auto it = m_cache.find(target); it != m_cache.end()) {
        return it->second;
    }

    const auto& node = NodeAccess::Node(target);
    if (node->const_val) {
        CpuTensor t;
        t.vtype = node->vtype;
        t.size = node->img_size;
        DRESSI_CHECK(t.size.isUniform(), "Constant must be {1,1}");
        t.data = *node->const_val;
        DRESSI_CHECK(t.data.size() == t.numComp(), "const_val size mismatch");
        m_cache[target] = t;
        return t;
    }

    Function creator = target.getCreator();
    DRESSI_CHECK(creator, "Unbound leaf Variable in CPU evaluation");
    const auto& func_node = NodeAccess::Node(creator);
    DRESSI_CHECK(func_node->cpu_func,
                 "Function '" + func_node->name + "' has no CPU kernel");

    std::vector<CpuTensor> inputs;
    for (const auto& x : creator.getInputVars()) {
        inputs.push_back(eval(x));
    }
    CpuTensor y = func_node->cpu_func(inputs);
    DRESSI_CHECK(y.vtype == target.getVType(),
                 "CPU kernel '" + func_node->name + "' vtype mismatch");
    DRESSI_CHECK(y.size == target.getImgSize(),
                 "CPU kernel '" + func_node->name + "' size mismatch");
    m_cache[target] = y;
    return y;
}

CpuTensor CpuTensorFromImage(const CpuImage& img, VType vtype) {
    DRESSI_CHECK(img.channels == NumComponents(vtype),
                 "CpuImage channel count does not match VType");
    CpuTensor t;
    t.vtype = vtype;
    t.size = {img.width, img.height};
    t.data = img.data;
    return t;
}

CpuImage CpuImageFromTensor(const CpuTensor& tensor) {
    CpuImage img(tensor.size.w, tensor.size.h, tensor.numComp());
    img.data = tensor.data;
    return img;
}

}  // namespace dressi
