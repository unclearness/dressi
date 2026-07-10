#include "core/op_builder.h"

#include "core/infer.h"

namespace dressi {

Variable MakeOp(OpDesc desc, const Variables& xs) {
    Function func(desc.fwd_code, desc.shader_type, desc.bwd);
    auto& node = *NodeAccess::Node(func);
    node.name = std::move(desc.name);
    node.cpu_func = std::move(desc.cpu);
    node.infer = std::move(desc.infer);
    node.input_access = std::move(desc.input_access);
    if (!node.input_access.empty()) {
        DRESSI_CHECK(node.input_access.size() == xs.size(),
                     "input_access size mismatch");
    }
    Variable y = func.buildFwd(xs);
    if (desc.const_val) {
        NodeAccess::Node(y)->const_val = std::move(desc.const_val);
    }
    return y;
}

CpuFunc ElementwiseCpu(CpuPixelKernel kernel, std::optional<VType> out_vtype) {
    return [kernel, out_vtype](const std::vector<CpuTensor>& xs) -> CpuTensor {
        DRESSI_CHECK(!xs.empty(), "Elementwise CPU kernel requires inputs");
        // Mirror InferElementwise for output shape
        ImgSize size = {1, 1};
        uint32_t ny = 1;
        VType vtype = xs[0].vtype;
        for (const auto& x : xs) {
            if (!x.size.isUniform()) {
                size = x.size;
            }
            if (x.numComp() > ny) {
                ny = x.numComp();
                vtype = x.vtype;
            }
        }
        if (out_vtype) {
            vtype = *out_vtype;
            ny = NumComponents(vtype);
        }

        CpuTensor y;
        y.vtype = vtype;
        y.size = size;
        y.data.resize(size_t(size.w) * size.h * ny);

        const size_t n_pixels = y.numPixels();
        std::vector<const float*> ptrs(xs.size());
        std::vector<uint32_t> ns(xs.size());
        std::vector<std::vector<float>> comp_bufs(xs.size());
        for (size_t i = 0; i < xs.size(); i++) {
            ns[i] = xs[i].numComp();
            comp_bufs[i].resize(ns[i]);
        }

        for (size_t p = 0; p < n_pixels; p++) {
            for (size_t i = 0; i < xs.size(); i++) {
                // {1,1} inputs broadcast spatially
                const size_t src_p = xs[i].size.isUniform() ? 0 : p;
                const float* src = xs[i].data.data() + src_p * ns[i];
                for (uint32_t c = 0; c < ns[i]; c++) {
                    comp_bufs[i][c] = src[c];
                }
                ptrs[i] = comp_bufs[i].data();
            }
            kernel(ptrs, ns, y.data.data() + p * ny, ny);
        }
        return y;
    };
}

namespace {

// Component read with scalar->vector broadcast (GLSL float op vecN semantics)
inline float CompAt(const float* x, uint32_t n, uint32_t c) {
    return x[n == 1 ? 0 : c];
}

}  // namespace

CpuFunc MapCpu1(float (*f)(float)) {
    return ElementwiseCpu([f](const std::vector<const float*>& xs,
                              const std::vector<uint32_t>& ns, float* y,
                              uint32_t ny) {
        for (uint32_t c = 0; c < ny; c++) {
            y[c] = f(CompAt(xs[0], ns[0], c));
        }
    });
}

CpuFunc MapCpu2(float (*f)(float, float)) {
    return ElementwiseCpu([f](const std::vector<const float*>& xs,
                              const std::vector<uint32_t>& ns, float* y,
                              uint32_t ny) {
        for (uint32_t c = 0; c < ny; c++) {
            y[c] = f(CompAt(xs[0], ns[0], c), CompAt(xs[1], ns[1], c));
        }
    });
}

CpuFunc MapCpu3(float (*f)(float, float, float)) {
    return ElementwiseCpu([f](const std::vector<const float*>& xs,
                              const std::vector<uint32_t>& ns, float* y,
                              uint32_t ny) {
        for (uint32_t c = 0; c < ny; c++) {
            y[c] = f(CompAt(xs[0], ns[0], c), CompAt(xs[1], ns[1], c),
                     CompAt(xs[2], ns[2], c));
        }
    });
}

InferFn FixedInfer(VType vtype) {
    return [vtype](const Variables& xs) -> std::pair<VType, ImgSize> {
        return {vtype, InferCommonImgSize(xs)};
    };
}

}  // namespace dressi
