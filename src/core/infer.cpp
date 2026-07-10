#include "core/infer.h"

#include <sstream>

namespace dressi {

ImgSize InferCommonImgSize(const Variables& xs) {
    ImgSize common = {1, 1};
    for (const auto& x : xs) {
        const ImgSize s = x.getImgSize();
        if (s.isUniform()) {
            continue;
        }
        if (common.isUniform()) {
            common = s;
        } else {
            DRESSI_CHECK(common == s,
                         "Image size mismatch: non-uniform inputs must have "
                         "the same size");
        }
    }
    return common;
}

std::pair<VType, ImgSize> InferElementwise(const Variables& xs) {
    DRESSI_CHECK(!xs.empty(), "Elementwise inference requires inputs");

    bool any_int = false;
    bool any_float = false;
    VType widest = FLOAT;
    uint32_t widest_n = 0;
    for (const auto& x : xs) {
        const VType t = x.getVType();
        (IsIntVType(t) ? any_int : any_float) = true;
        const uint32_t n = NumComponents(t);
        if (n > 1) {
            // Scalars broadcast; vectors/matrices must agree with each other
            DRESSI_CHECK(widest_n <= 1 || t == widest,
                         "Vector/matrix type mismatch between inputs");
        }
        if (n > widest_n) {
            widest_n = n;
            widest = t;
        }
    }
    DRESSI_CHECK(!(any_int && any_float),
                 "Int and float inputs cannot mix implicitly; use a cast");

    return {widest, InferCommonImgSize(xs)};
}

std::pair<VType, ImgSize> InferOutput(const FunctionNode& func,
                                      const Variables& xs) {
    if (func.infer) {
        return func.infer(xs);
    }
    return InferElementwise(xs);
}

}  // namespace dressi
