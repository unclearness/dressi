#ifndef DRESSI_EXAMPLES_BENCH_H
#define DRESSI_EXAMPLES_BENCH_H

#include <cstdint>
#include <string>
#include <vector>

namespace dressi {
class DressiAD;
}

namespace dressi_examples {

// The run's viewer condition (docs/benchmarks.md, "Measurement conditions"),
// from what actually happened rather than from the flags: `viewer_open` is
// false for --no-view and for a host without a display. "partial" is a live
// viewer updated less often than every iteration, which is not a benchmark
// condition.
std::string ViewerCondition(bool viewer_open, int view_interval);

// Median of per-iteration wall-time samples (0 when empty). Steady-state
// median with the warmup excluded is THE cross-device metric — see
// CLAUDE.md "Benchmarking".
double MedianMs(std::vector<double> samples);

// One-time build/warmup overhead in ms: the extra wall time the excluded
// warmup iterations cost beyond the steady-state rate. Captures the full
// pack + GLSL-compile + Vulkan-build ladder AND the reactive-cache
// rebuilds at iter ~2 and ~8 (all within the warmup window).
// `warmup_samples` are the per-iter ms of the excluded warmup iters;
// `steady_median` is the post-warmup MedianMs. Rough by design (wall
// clock, not the per-phase breakdown) — see CLAUDE.md "Benchmarking".
double WarmupMs(const std::vector<double>& warmup_samples,
                double steady_median);

// One flat-JSON benchmark record for scripts/bench_summary.py. The
// constructor fills the common keys (example, device, platform); add the
// example's parameters and metrics, then save() into the run's out dir.
class BenchRecord {
public:
    BenchRecord(const std::string& example, const std::string& device);

    void add(const std::string& key, const std::string& value);
    void add(const std::string& key, double value, int precision = 4);
    void add(const std::string& key, int64_t value);
    void add(const std::string& key, bool value);

    // Records what the figure depends on: the viewer condition, the timed
    // scope (execStep, send+execStep, or send+execStep+recv), build type,
    // commit, driver version, packing mode, and the packing reduction as three
    // keys (funcs, substages, stages): the unpacked backward-graph op count
    // collapsed by substage packing then by stage packing into Vulkan render
    // passes. Call it after the steady-state build. The packed counts are
    // DEVICE-DEPENDENT — greedy fusion is bounded by the physical device's
    // Vulkan limits, so the same graph packs differently per GPU.
    void addRun(const dressi::DressiAD& ad, const std::string& viewer,
                const std::string& timed_scope);

    void save(const std::string& path) const;

private:
    std::string m_json;  // accumulated "key":value pairs
};

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_BENCH_H
