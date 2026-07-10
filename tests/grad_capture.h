// Shared GPU-test helper: an identity optimizer whose lambda captures the
// backward-graph gradient Variables and marks them as outputs so they can
// be read back with recvImg (the CPU-side vertex-update pattern).

#ifndef DRESSI_TESTS_GRAD_CAPTURE_H
#define DRESSI_TESTS_GRAD_CAPTURE_H

#include "dressi/dressi.h"

namespace dressi_test {

struct GradCapture {
    dressi::Variables xs, gxs;
    bool marked = false;
};

inline dressi::DressiAD::Optimizer IdentityOptimizer(dressi::DressiAD& ad,
                                                     GradCapture& cap) {
    using namespace dressi;
    return [&ad, &cap](Variables xs, Variables gxs) {
        Variables updated;
        for (size_t i = 0; i < xs.size(); i++) {
            updated.push_back(xs[i] + F::Float(0.f) * gxs[i]);
        }
        if (!cap.marked) {
            cap.xs = xs;
            cap.gxs = gxs;
            for (const auto& g : gxs) {
                ad.markOutput(g);
            }
            cap.marked = true;
        }
        return updated;
    };
}

inline dressi::Variable GradOf(const GradCapture& cap,
                               const dressi::Variable& wrt) {
    for (size_t i = 0; i < cap.xs.size(); i++) {
        if (cap.xs[i] == wrt) {
            return cap.gxs[i];
        }
    }
    return dressi::Variable(nullptr);
}

}  // namespace dressi_test

#endif  // DRESSI_TESTS_GRAD_CAPTURE_H
