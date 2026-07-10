#ifndef DRESSI_CORE_INFER_H
#define DRESSI_CORE_INFER_H

#include <utility>

#include "core/node.h"

namespace dressi {

// Default component-wise inference:
// - ImgSize: all non-{1,1} inputs must agree; {1,1} broadcasts.
// - VType: FLOAT broadcasts against VECn; equal vector types pass through;
//   int and float never mix implicitly.
std::pair<VType, ImgSize> InferElementwise(const Variables& xs);

// Dispatches to the FunctionNode's custom InferFn if present (stored via
// OpBuilder), otherwise InferElementwise.
std::pair<VType, ImgSize> InferOutput(const FunctionNode& func,
                                      const Variables& xs);

ImgSize InferCommonImgSize(const Variables& xs);

}  // namespace dressi

#endif  // DRESSI_CORE_INFER_H
