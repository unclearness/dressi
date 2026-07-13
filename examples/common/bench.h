#ifndef DRESSI_EXAMPLES_BENCH_H
#define DRESSI_EXAMPLES_BENCH_H

#include <cstdint>
#include <string>
#include <vector>

namespace dressi_examples {

// Median of per-iteration wall-time samples (0 when empty). Steady-state
// median with the warmup excluded is THE cross-device metric — see
// CLAUDE.md "Benchmarking".
double MedianMs(std::vector<double> samples);

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

    void save(const std::string& path) const;

private:
    std::string m_json;  // accumulated "key":value pairs
};

}  // namespace dressi_examples

#endif  // DRESSI_EXAMPLES_BENCH_H
