#!/usr/bin/env python3
"""Summarize cross-device benchmark records into Markdown tables.

texture_optimization and silhouette_optimization write a one-object
``bench.json`` into their ``--out-dir`` at the end of a run (fields:
device, platform, parameters, median steady-state ms/iter, quality).
Collect those files from every machine/phone and render one table per
example, one row per run, sorted by median time.

Usage:
    python scripts/bench_summary.py [path ...]

Each path may be a bench*.json file or a directory (searched
recursively; build trees and .git are skipped). With no arguments the
current directory is searched — e.g. after:

    # desktop runs write ./texopt_out/bench.json etc. in place
    ./build/examples/Release/texture_optimization.exe data/bunny
    ./build/examples/Release/silhouette_optimization.exe data/bunny --no-view=1

    # pull one file per example per Android device (any filename works)
    adb shell run-as org.dressi.examples cat files/out/texture_optimization/bench.json > bench_results/pixel_texopt.json
    adb shell run-as org.dressi.examples cat files/out/silhouette_optimization/bench.json > bench_results/pixel_sil.json

    python scripts/bench_summary.py texopt_out silopt_out_hardsoftras bench_results
"""

import json
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
    if rec.get("example") == "silhouette_optimization":
        p = (f"{rec.get('technique', '?')} {rec.get('screen', '?')}px "
             f"{rec.get('views', '?')}v {rec.get('iters', '?')}it "
             f"s{rec.get('samples', '?')} K{rec.get('peels', '?')}")
        if rec.get("single_view"):
            p += " single-view"
        return p
    return (f"{rec.get('screen', '?')}px {rec.get('views', '?')}v "
            f"{rec.get('iters', '?')}it")


def quality_of(rec):
    if "mean_iou" in rec:
        return f"IoU {rec['mean_iou']:.4f}"
    if "accurate_pct" in rec:
        return f"{rec['accurate_pct']:.1f}% accurate"
    return ""


def main():
    records = find_records(sys.argv[1:] or ["."])
    if not records:
        print("no bench*.json records found", file=sys.stderr)
        return 1
    for example in sorted({r.get("example", "?") for r in records}):
        rows = [r for r in records if r.get("example") == example]
        rows.sort(key=lambda r: r.get("median_ms_per_iter", float("inf")))
        print(f"## {example}\n")
        print("| device | platform | parameters | median ms/iter | quality |")
        print("| --- | --- | --- | ---: | --- |")
        for r in rows:
            print(f"| {r.get('device', '?')} | {r.get('platform', '?')} "
                  f"| {params_of(r)} "
                  f"| {r.get('median_ms_per_iter', float('nan')):.3f} "
                  f"| {quality_of(r)} |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
