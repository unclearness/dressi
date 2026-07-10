#ifndef DRESSI_CORE_BUILD_BACKWARD_H
#define DRESSI_CORE_BUILD_BACKWARD_H

#include <tuple>

#include "dressi/function.h"
#include "dressi/variable.h"

namespace dressi {

// Traverses the forward computational graph backward from `loss_var`,
// sums up gradient variables by the chain rule, and returns the pair of
// requires-grad leaf variables and their gradient variables (index-aligned,
// ordered by creation id for stable optimizer pairing).
// `loss_var` must be a scalar: FLOAT {1,1}.
std::tuple<Variables, Variables> BuildBackward(const Variable& loss_var);

// Reduces a gradient to the shape of its target variable: spatial sum when
// the target is {1,1} but the gradient is an image (uniform broadcast), and
// component sum when the target is scalar but the gradient is a vector
// (scalar-to-vector broadcast).
Variable SumToShape(const Variable& grad, const Variable& target);

// Collects all functions reachable from `roots` through creators, sorted by
// creation id (a valid topological order).
Functions TraverseFuncs(const Variables& roots);

}  // namespace dressi

#endif  // DRESSI_CORE_BUILD_BACKWARD_H
