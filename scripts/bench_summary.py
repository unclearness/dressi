#!/usr/bin/env python3
"""Summarize cross-device benchmark records into Markdown tables.

Every example writes a one-object ``bench.json`` into its ``--out-dir``
at the end of a run: GPU device name, host info (os / cpu / ram), the
parameters, the MEDIAN steady-state ms per iteration (warmup excluded),
the one-time build/warmup overhead in ms (pack + GLSL compile + Vulkan
build + reactive rebuilds; ``first_build_ms`` is iter 0 alone), and the
example's quality metric. Collect those files from every
machine/phone and render one table per example and condition, sorted by
median time. Repeated runs of one configuration collapse into one row: the
median of the run medians, with the min-max range (docs/benchmarks.md,
"Measurement conditions"). Records with a different viewer condition or timed
scope never share a table; records from before those keys existed are
grouped as "unknown".

Usage:
    python scripts/bench_summary.py [path ...]

Each path may be a bench*.json file or a directory (searched
recursively; build trees and .git are skipped). With no arguments the
current directory is searched — e.g. after:

    python scripts/run_benches.py -o bench_results/desktop
    python scripts/run_benches.py --target android -o bench_results/pixel8

    python scripts/bench_summary.py bench_results
"""

import json
import statistics
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".cxx", ".gradle", "_deps", "node_modules"}


def find_records(paths):
    files = []
    for p in map(Path, paths):
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in p.rglob("bench*.json"):
                parts = set(f.parts)
                if parts & SKIP_DIRS or any(
                        part.startswith("build") for part in f.parts):
                    continue
                files.append(f)
    records = []
    for f in sorted(set(files)):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["_file"] = str(f)
            records.append(rec)
        except (json.JSONDecodeError, OSError) as e:
            print(f"warning: skipping {f}: {e}", file=sys.stderr)
    return records


def params_of(rec):
    ex = rec.get("example")
    if ex == "silhouette_optimization":
        p = (f"{rec.get('technique', '?')} {rec.get('screen', '?')}px "
             f"{rec.get('views', '?')}v {rec.get('iters', '?')}it "
             f"s{rec.get('samples', '?')} K{rec.get('peels', '?')}")
        if rec.get("single_view"):
            p += " single-view"
        return p
    if ex == "texture_optimization":
        return (f"{rec.get('screen', '?')}px {rec.get('views', '?')}v "
                f"{rec.get('iters', '?')}it")
    if ex == "image_fitting":
        return f"{rec.get('screen', '?')}px {rec.get('iters', '?')}it"
    if ex == "pbr_material_optimization":
        return (f"{rec.get('optimize', '?')} {rec.get('screen', '?')}px "
                f"tex{rec.get('tex', '?')} {rec.get('views', '?')}v "
                f"{rec.get('iters', '?')}it")
    if ex == "pbr_envmap_optimization":
        p = (f"{rec.get('screen', '?')}px env{rec.get('envres', '?')} "
             f"{rec.get('views', '?')}v {rec.get('iters', '?')}it")
        if "fg_only" in rec:
            p += f" fg_only={int(rec['fg_only'])}"
        return p
    if ex == "pbr_shading":
        return f"{rec.get('screen', '?')}px {rec.get('frames', '?')} frames"
    return ""


def warmup_of(runs):
    # One-time build/warmup overhead (ms), with the single largest build
    # (iter 0) in parentheses. "-" for older records without the field.
    # Each is the median across runs, separately from the ms/iter median: the
    # first run after an install can be a cold start, 6.7x longer on the A14.
    def median_of(key):
        values = [r[key] for r in runs if key in r]
        return statistics.median_low(values) if values else None

    warmup = median_of("warmup_ms")
    if warmup is None:
        return "-"
    s = f"{warmup:.0f}"
    first = median_of("first_build_ms")
    if first is not None:
        s += f" ({first:.0f})"
    return s


def packing_of(rec):
    # Packing reduction "funcs->substages->stages": the unpacked backward-graph
    # op count collapsed by substage packing then by stage packing into Vulkan
    # render passes. Device-dependent (greedy fusion is bounded by the GPU's
    # Vulkan limits). "-" for older records without the fields.
    if "stages" not in rec:
        return "-"
    funcs = rec.get("funcs")
    substages = rec.get("substages")
    stages = rec["stages"]
    if funcs is not None and substages is not None:
        return f"{funcs}->{substages}->{stages}"
    return str(stages)


def quality_of(rec, median_ms):
    if "mean_iou" in rec:
        return f"IoU {rec['mean_iou']:.4f}"
    if "accurate_pct" in rec:
        return f"{rec['accurate_pct']:.1f}% accurate"
    if "psnr_db" in rec:
        return f"PSNR {rec['psnr_db']:.2f} dB"
    if "fps" in rec:
        return f"{1000.0 / median_ms:.0f} FPS"
    if "max_err" in rec:
        return f"max err {rec['max_err']:.4f}"
    return ""


def host_of(rec):
    parts = []
    if rec.get("cpu"):
        cpu = rec["cpu"]
        if rec.get("cpu_cores"):
            cpu += f" ({rec['cpu_cores']}c)"
        parts.append(cpu)
    if rec.get("ram_gb"):
        parts.append(f"{rec['ram_gb']:.0f} GB")
    if rec.get("os"):
        parts.append(rec["os"])
    return ", ".join(parts) or rec.get("platform", "")


def group_of(rec):
    # One row: a configuration on one device from one build. Two builds or
    # drivers stay on separate rows even under the same condition.
    return (rec.get("device", "?"), rec.get("platform", "?"), params_of(rec),
            rec.get("build_type", "?"), rec.get("git_commit", "?"),
            rec.get("driver_version", "?"), rec.get("packing_mode", "?"))


def row_of(runs):
    medians = [r.get("median_ms_per_iter", float("nan")) for r in runs]
    # median_low: an even run count must not average two run medians
    median = statistics.median_low(medians)
    # Packing and quality are shown for the run behind the median
    rec = runs[medians.index(median)]
    return median, min(medians), max(medians), len(runs), rec, warmup_of(runs)


def print_table(example, viewer, scope, rows):
    print(f"## {example} (viewer: {viewer}, timed scope: {scope})\n")
    print("| device | host | parameters | median ms/iter | min-max | runs "
          "| warmup ms (first) | packing f->s->st | quality | commit |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    groups = {}
    for r in rows:
        groups.setdefault(group_of(r), []).append(r)
    for median, low, high, n, r, warmup in sorted(map(row_of, groups.values()),
                                                  key=lambda row: row[0]):
        print(f"| {r.get('device', '?')} | {host_of(r)} "
              f"| {params_of(r)} "
              f"| {median:.3f} "
              f"| {low:.3f}-{high:.3f} "
              f"| {n} "
              f"| {warmup} "
              f"| {packing_of(r)} "
              f"| {quality_of(r, median)} "
              f"| {r.get('git_commit', '?')} |")
    print()


def main():
    records = find_records(sys.argv[1:] or ["."])
    if not records:
        print("no bench*.json records found", file=sys.stderr)
        return 1
    tables = {}
    for r in records:
        key = (r.get("example", "?"), r.get("viewer", "unknown"),
               r.get("timed_scope", "unknown"))
        tables.setdefault(key, []).append(r)
    for key in sorted(tables):
        print_table(*key, tables[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
