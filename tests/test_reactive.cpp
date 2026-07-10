// Reactive cache tests: pass-count shrinkage after FAST/FULL rebuilds,
// result consistency with caching enabled vs disabled, and re-expansion
// when a pruned branch's input changes.

#include <gtest/gtest.h>

#include <cmath>

#include "dressi/dressi.h"

using namespace dressi;

namespace {

CpuImage MakeImage(uint32_t w, uint32_t h, uint32_t c,
                   const std::function<float(size_t)>& gen) {
    CpuImage img(w, h, c);
    for (size_t i = 0; i < img.data.size(); i++) {
        img.data[i] = gen(i);
    }
    return img;
}

// A graph with a heavy static branch: the target preprocessing (reduction
// chain) depends only on the target leaf and becomes prunable once cached.
struct ReactiveGraph {
    Variable param{VEC3, {16, 16}};
    Variable target{VEC3, {16, 16}};
    Variable loss{nullptr};

    ReactiveGraph() {
        Variable t_proc = F::Exp(target * 0.5f);
        Variable t_scale = F::Mean(t_proc);  // static reduction chain
        Variable diff = t_proc - param * (t_scale + 0.5f);
        loss = F::Mean(diff * diff);
        Variable param_mut = param;
        param_mut.setRequiresGradRecursively();
    }

    void wire(DressiAD& ad, float lr) const {
        ad.setLossVar(loss);
        ad.setOptimizer([lr](Variables xs, Variables gxs) {
            Variables updated;
            for (size_t i = 0; i < xs.size(); i++) {
                updated.push_back(xs[i] - gxs[i] * lr);
            }
            return updated;
        });
    }
};

}  // namespace

TEST(Reactive, PassCountShrinksAfterRebuild) {
    ReactiveGraph s;
    DressiAD ad;
    s.wire(ad, 0.1f);
    ad.setRebuildCounts(2, 6);

    ad.sendImg(s.param, MakeImage(16, 16, 3, [](size_t) { return 0.4f; }));
    ad.sendImg(s.target, MakeImage(16, 16, 3, [](size_t i) {
                   return 0.2f + 0.01f * float(i % 29);
               }));

    ad.execStep();
    const size_t full_substages = ad.getSubStageCount();
    ASSERT_GT(full_substages, 0u);

    // Iterate without new uploads until past both rebuild thresholds
    for (int i = 0; i < 10; i++) {
        ad.execStep();
    }
    const size_t pruned_substages = ad.getSubStageCount();
    EXPECT_LT(pruned_substages, full_substages)
            << "static target branch was not pruned";
}

TEST(Reactive, ResultsMatchWithCachingDisabled) {
    ReactiveGraph s1;
    ReactiveGraph s2;

    DressiAD cached;
    s1.wire(cached, 0.1f);
    cached.setRebuildCounts(2, 6);

    DressiAD uncached;
    s2.wire(uncached, 0.1f);
    uncached.setRebuildCounts(0, 0);  // never prune

    const auto init_p = [](size_t) { return 0.4f; };
    const auto init_t = [](size_t i) { return 0.2f + 0.01f * float(i % 29); };
    cached.sendImg(s1.param, MakeImage(16, 16, 3, init_p));
    cached.sendImg(s1.target, MakeImage(16, 16, 3, init_t));
    uncached.sendImg(s2.param, MakeImage(16, 16, 3, init_p));
    uncached.sendImg(s2.target, MakeImage(16, 16, 3, init_t));

    for (int i = 0; i < 12; i++) {
        cached.execStep();
        uncached.execStep();
    }
    const CpuImage a = cached.recvImg(s1.loss);
    const CpuImage b = uncached.recvImg(s2.loss);
    EXPECT_NEAR(a.data[0], b.data[0], 1e-5f);

    const CpuImage pa = cached.recvImg(s1.param);
    const CpuImage pb = uncached.recvImg(s2.param);
    for (size_t i = 0; i < pa.data.size(); i++) {
        ASSERT_NEAR(pa.data[i], pb.data[i], 1e-5f) << "index " << i;
    }
}

TEST(Reactive, DirtyChangeReexpandsPrunedPlan) {
    ReactiveGraph s;
    DressiAD ad;
    s.wire(ad, 0.0f);  // keep parameters fixed to isolate the target effect
    ad.setRebuildCounts(2, 6);

    ad.sendImg(s.param, MakeImage(16, 16, 3, [](size_t) { return 0.4f; }));
    ad.sendImg(s.target, MakeImage(16, 16, 3, [](size_t) { return 0.2f; }));

    ad.execStep();
    const size_t full_substages = ad.getSubStageCount();
    for (int i = 0; i < 8; i++) {
        ad.execStep();
    }
    const size_t pruned_substages = ad.getSubStageCount();
    ASSERT_LT(pruned_substages, full_substages);
    const float loss_before = ad.recvImg(s.loss).data[0];

    // Changing the target must re-expand the pruned target branch (constant
    // gradient-prefix functions may legitimately stay cached, so the count
    // does not return all the way to `full_substages`) and change the loss
    ad.sendImg(s.target, MakeImage(16, 16, 3, [](size_t) { return 0.9f; }));
    ad.execStep();
    EXPECT_GT(ad.getSubStageCount(), pruned_substages)
            << "pruned plan was not re-expanded";
    const float loss_after = ad.recvImg(s.loss).data[0];
    EXPECT_GT(std::abs(loss_after - loss_before), 1e-3f);
}
