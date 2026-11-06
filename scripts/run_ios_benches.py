#!/usr/bin/env python3
"""Run the example benchmarks on an iPhone and pull one bench.json per run.

Needs DressiExamples installed (README-ios.md) and the device unlocked. Each run
launches the app with `--run=<entry> --no-view=1`, copies its bench.json
out of the app container, and rests so the phone cools down. The app exits
between runs, so the home screen is normal for most of the time this takes.

Usage:
    python3 scripts/run_ios_benches.py --device <udid> [-o out_root]
        [--reps 3] [--only label,label] [--cool 90] [--avocado]
    python3 scripts/bench_summary.py bench_results/ios

--avocado adds the one-view Avocado resolution sweep (docs/benchmarks.md) under
<out_root>/avocado/.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

BUNDLE_ID = "org.dressi.examples"
# entry, timeout in seconds
ENTRIES = [
    ("image_fitting", 120),
    ("texture_optimization", 400),
    ("silhouette_optimization_hardsoftras", 200),
    ("silhouette_optimization_aa", 200),
    ("pbr_shading", 120),
    ("pbr_material_optimization", 400),
    ("pbr_envmap_optimization", 1100),
]
HSR = "silhouette_optimization_hardsoftras"
AA = "silhouette_optimization_aa"
SWEEP = [
    ("hsr_k1", HSR, ["--peels=1"]),
    ("hsr_k3", HSR, ["--peels=3"]),
    ("aa", AA, []),
]
RESOLUTIONS = (256, 512, 1024, 2048, 4096)


def devicectl(*args, **kwargs):
    return subprocess.run(["xcrun", "devicectl", "device", *args], **kwargs)


def bundle_path(device):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "apps.json"
        devicectl("info", "apps", "--device", device, "--bundle-id", BUNDLE_ID,
                  "--json-output", str(out), capture_output=True, check=True)
        app = json.loads(out.read_text())["result"]["apps"][0]
    return app["url"].replace("file://", "").rstrip("/")


def run(device, root, entry, tag, extra, timeout):
    log = root / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        with log.open("w") as f:
            code = devicectl(
                    "process", "launch", "--console", "--terminate-existing",
                    "--device", device, BUNDLE_ID, f"--run={entry}",
                    "--no-view=1", *extra,
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    dst = root / tag
    shutil.rmtree(dst, ignore_errors=True)
    # A failed run leaves the previous run's bench.json in the container.
    if code == 0:
        dst.mkdir(parents=True)
        devicectl("copy", "from", "--device", device,
                  "--domain-type", "appDataContainer",
                  "--domain-identifier", BUNDLE_ID,
                  "--source", f"Documents/out/{entry}/bench.json",
                  "--destination", str(dst / "bench.json"),
                  capture_output=True, check=True)
    print(f"{tag}: exit {code}, {time.time() - start:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", required=True,
                   help="udid from `xcrun devicectl list devices`")
    p.add_argument("-o", "--out-root", default="bench_results/ios")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--only", default="",
                   help="comma-separated entry substrings to keep")
    p.add_argument("--cool", type=int, default=90,
                   help="seconds to rest after each run")
    p.add_argument("--avocado", action="store_true")
    args = p.parse_args()

    root = Path(args.out_root)
    pats = [s for s in args.only.split(",") if s]
    for rep in range(1, args.reps + 1):
        for entry, timeout in ENTRIES:
            if pats and not any(s in entry for s in pats):
                continue
            run(args.device, root, entry, f"run{rep}/{entry}", [], timeout)
            time.sleep(args.cool)

    if args.avocado:
        mesh = bundle_path(args.device) + "/data/Avocado/glTF/Avocado.gltf"
        for res in RESOLUTIONS:
            for label, entry, opts in SWEEP:
                for rep in range(1, args.reps + 1):
                    run(args.device, root, entry,
                        f"avocado/{label}_{res}_r{rep}",
                        [f"--mesh={mesh}", "--views=1", f"--screen={res}",
                         "--iters=300", *opts], 400)
                    time.sleep(30)


if __name__ == "__main__":
    main()
