#!/usr/bin/env bash
# Run every example's benchmark into one output root, one bench.json per run,
# under the conditions in docs/benchmarks.md ("Measurement conditions").
#
# A thin wrapper that just *executes* the example exes with headless flags and
# per-example --out-dir. It deliberately does NOT summarize: aggregate the
# results separately with
#
#     python3 scripts/bench_summary.py <out-root>
#
# so running and reporting stay decoupled (re-summarize old runs, mix in
# Android bench.json files pulled with adb, etc. without re-running). Each
# bench.json is a MEDIAN steady-state execStep record (warmup excluded) written
# by the example itself via examples/common/bench.h — see the "Benchmarking"
# section of CLAUDE.md for the methodology these encode.
#
# Usage:
#   scripts/run_benches.sh [-o out_root] [-b build_dir] [-d data_dir]
#                          [-n label,label] [-r reps] [-l]
#   -o  output root (default: bench_results)
#   -b  dir holding the example exes (default: build/examples/Release)
#   -d  dataset root for the mesh examples (default: data)
#   -n  comma-separated label substrings; only matching benchmarks run
#   -r  runs per benchmark, into <out_root>/run<k>/<label> (default: 3)
#   -l  list the benchmark commands and exit
#
# Example:
#   scripts/run_benches.sh
#   python3 scripts/bench_summary.py bench_results
set -u

out_root='bench_results'
build_dir='build/examples/Release'
data_dir='data'
only=''
reps=3
list_only=0

while getopts 'o:b:d:n:r:lh' opt; do
    case "$opt" in
        o) out_root=$OPTARG ;;
        b) build_dir=$OPTARG ;;
        d) data_dir=$OPTARG ;;
        n) only=$OPTARG ;;
        r) reps=$OPTARG ;;
        l) list_only=1 ;;
        h|*) sed -n '2,30p' "$0"; exit 0 ;;
    esac
done

# Resolve the repo root (this script lives in scripts/) and cd there.
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo" || exit 1

# Benchmark table: "label|exe|space-separated headless args". Each run appends
# its own --out-dir=<out_root>/<label>. Edit here to add/change coverage.
benches=(
    "image_fitting|image_fitting|--no-view=1"
    "texture_optimization|texture_optimization|$data_dir/bunny --no-view=1"
    "silhouette_hardsoftras|silhouette_optimization|$data_dir/bunny --technique=hardsoftras --no-view=1"
    "silhouette_aa|silhouette_optimization|$data_dir/bunny --technique=aa --no-view=1"
    "pbr_material_optimization|pbr_material_optimization|--no-view=1"
    "pbr_envmap_optimization|pbr_envmap_optimization|--no-view=1"
    "pbr_shading|pbr_shading|--frames=200 --no-view=1"
)

# Optional label filter (-n foo,bar keeps labels containing foo or bar).
match_only() {
    [ -z "$only" ] && return 0
    local label=$1 pat
    IFS=',' read -ra pats <<< "$only"
    for pat in "${pats[@]}"; do
        [ -n "$pat" ] && case "$label" in *"$pat"*) return 0 ;; esac
    done
    return 1
}

if [ "$list_only" -eq 1 ]; then
    for row in "${benches[@]}"; do
        IFS='|' read -r label exe args <<< "$row"
        match_only "$label" || continue
        printf '%-28s %s/%s %s --out-dir=%s/run<k>/%s\n' \
            "$label" "$build_dir" "$exe" "$args" "$out_root" "$label"
    done
    exit 0
fi

mkdir -p "$out_root"
# GPU state during each run (docs/benchmarks.md, "Device state"): some drivers
# keep a GPU with no window in a power-saving state. bench_summary.py checks it.
have_smi=0
command -v nvidia-smi >/dev/null 2>&1 && have_smi=1

fail=0
declare -a summary
for rep in $(seq 1 "$reps"); do
for row in "${benches[@]}"; do
    IFS='|' read -r label exe args <<< "$row"
    match_only "$label" || continue

    bin="$build_dir/$exe"
    [ -x "$bin" ] || bin="$build_dir/$exe.exe"   # tolerate a Windows layout
    if [ ! -x "$bin" ]; then
        echo "WARNING: $label: missing $build_dir/$exe (build first) - skipped" >&2
        summary+=("run$rep/$label	MISSING")
        continue
    fi
    out="$out_root/run$rep/$label"
    # A failed run must not leave an older record behind
    rm -f "$out/bench.json"

    rm -f "$out/gpu_state.csv"
    sampler=''
    if [ "$have_smi" -eq 1 ]; then
        mkdir -p "$out"
        nvidia-smi --query-gpu=timestamp,name,pstate,clocks.sm,clocks.max.sm,power.draw \
            --format=csv -lms 200 > "$out/gpu_state.csv" 2>/dev/null &
        sampler=$!
    fi

    # shellcheck disable=SC2086  # args is an intentional word-split arg string
    echo "==> run$rep/$label: $bin $args --out-dir=$out"
    # shellcheck disable=SC2086
    "$bin" $args "--out-dir=$out"
    code=$?
    if [ -n "$sampler" ]; then
        kill "$sampler" 2>/dev/null
        wait "$sampler" 2>/dev/null
    fi

    if [ "$code" -eq 0 ]; then
        summary+=("run$rep/$label	OK")
    else
        summary+=("run$rep/$label	EXIT $code")
        echo "WARNING: $label exited with code $code" >&2
        fail=1
    fi
done
done

echo
echo '=== bench run summary ==='
printf '%s\n' "${summary[@]}" | column -t -s "$(printf '\t')" 2>/dev/null \
    || printf '%s\n' "${summary[@]}"
echo "Aggregate with:  python3 scripts/bench_summary.py $out_root"

exit "$fail"
