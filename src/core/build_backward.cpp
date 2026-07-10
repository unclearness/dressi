#include "core/build_backward.h"

#include <algorithm>
#include <map>
#include <queue>
#include <unordered_set>

#include "core/node.h"
#include "dressi/f.h"

namespace dressi {

Variable SumToShape(const Variable& grad, const Variable& target) {
    Variable g = grad;
    const uint32_t n_target = NumComponents(target.getVType());
    const uint32_t n_grad = NumComponents(g.getVType());
    if (n_grad != n_target) {
        DRESSI_CHECK(n_target == 1,
                     "Gradient component count exceeds target and target is "
                     "not scalar");
        g = F::CompSum(g);
    }
    if (target.getImgSize().isUniform() && !g.getImgSize().isUniform()) {
        g = F::Sum(g);
    } else if (!target.getImgSize().isUniform() &&
               g.getImgSize().isUniform()) {
        // A uniform gradient reaching an image input (e.g. the {1,1} loss
        // seed passing through identity backwards): replicate per pixel
        g = F::Broadcast(g, target.getImgSize());
    }
    DRESSI_CHECK(g.getImgSize() == target.getImgSize(),
                 "Gradient image size does not reduce to target size");
    return g;
}

namespace {

// Sums gradient contributions as a tree of small SumPixelWise nodes: a
// single n-ary node needs n input attachments, and Vulkan only guarantees
// maxPerStageDescriptorInputAttachments >= 4
Variable SumContribs(Variables contribs) {
    constexpr size_t kMaxArity = 4;
    while (contribs.size() > 1) {
        Variables next;
        for (size_t i = 0; i < contribs.size(); i += kMaxArity) {
            const size_t n = std::min(kMaxArity, contribs.size() - i);
            if (n == 1) {
                next.push_back(contribs[i]);
            } else {
                next.push_back(F::SumPixelWise(
                        Variables(contribs.begin() + int64_t(i),
                                  contribs.begin() + int64_t(i + n))));
            }
        }
        contribs = std::move(next);
    }
    return contribs[0];
}

}  // namespace

std::tuple<Variables, Variables> BuildBackward(const Variable& loss_var) {
    DRESSI_CHECK(loss_var, "Null loss Variable");
    // The loss may be any float image (per the paper): seeding every pixel
    // with gradient 1 differentiates the implicit sum over pixels and
    // components, so no forward reduction chain to {1,1} is needed. The
    // {1,1} seed constant reaches image size through uniform broadcasting.
    DRESSI_CHECK(!IsIntVType(loss_var.getVType()) &&
                         !IsMatrixVType(loss_var.getVType()),
                 "Loss must be a float scalar/vector Variable");

    // Mapping from a forward variable to its gradient contributions
    std::map<Variable, Variables> fwd_bwds_map;

    // Pops the latest-created function first. Creation order is a valid
    // topological order (a function only consumes older variables), so all
    // consumers of a function's output are processed before the function
    // itself and its gradient sum is complete at pop time.
    const auto by_id = [](const Function& a, const Function& b) {
        return a.id() < b.id();
    };
    std::priority_queue<Function, std::vector<Function>, decltype(by_id)>
            func_queue(by_id);
    std::unordered_set<uint64_t> seen_funcs;

    // Set the loss as the starting point
    fwd_bwds_map[loss_var] = {F::Float(1.f)};
    if (Function creator = loss_var.getCreator()) {
        func_queue.push(creator);
    }

    while (!func_queue.empty()) {
        Function func = func_queue.top();
        func_queue.pop();
        if (!seen_funcs.insert(func.id()).second) {
            continue;
        }

        // Sum up the gradient contributions of the output
        Variable y = func.getOutputVar();
        auto it = fwd_bwds_map.find(y);
        if (it == fwd_bwds_map.end()) {
            continue;  // no gradient path reached this function
        }
        Variable gy = SumContribs(it->second);
        fwd_bwds_map.erase(it);

        const Variables xs = func.getInputVars();
        for (uint32_t x_idx = 0; x_idx < xs.size(); x_idx++) {
            const Variable& x = xs[x_idx];
            if (!x.getRequiresGrad()) {
                continue;  // skip no-gradient paths
            }
            Variable gx = func.buildBwd(gy, x_idx);
            if (!gx) {
                continue;  // non-backwardable function
            }
            gx = SumToShape(gx, x);
            fwd_bwds_map[x].push_back(gx);
            if (Function x_creator = x.getCreator()) {
                func_queue.push(x_creator);
            }
        }
    }

    // Remaining entries are the requires-grad leaves (no processed creator)
    Variables input_vars, input_grad_vars;
    for (auto& [x, gxs] : fwd_bwds_map) {
        if (x == loss_var) {
            continue;  // untouched seed (loss had no creator)
        }
        input_vars.push_back(x);
        input_grad_vars.push_back(SumToShape(SumContribs(gxs), x));
    }
    return {input_vars, input_grad_vars};
}

Functions TraverseFuncs(const Variables& roots) {
    Functions funcs;
    std::unordered_set<uint64_t> visited_funcs;
    std::unordered_set<uint64_t> visited_vars;
    std::vector<Variable> stack;
    for (const auto& root : roots) {
        if (root) {
            stack.push_back(root);
        }
    }
    while (!stack.empty()) {
        Variable var = stack.back();
        stack.pop_back();
        if (!visited_vars.insert(var.id()).second) {
            continue;
        }
        Function creator = var.getCreator();
        if (!creator || !visited_funcs.insert(creator.id()).second) {
            continue;
        }
        funcs.push_back(creator);
        for (const auto& x : creator.getInputVars()) {
            stack.push_back(x);
        }
    }
    std::sort(funcs.begin(), funcs.end(),
              [](const Function& a, const Function& b) {
                  return a.id() < b.id();
              });
    return funcs;
}

}  // namespace dressi
