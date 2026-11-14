# Benchmarks and build docs for macOS and iOS

## Goal

Measure the README and docs/benchmarks.md figures on macOS and iOS, add them,
update the surrounding text, and document how to build on both.

## What was done

- Added Apple M5 Pro and Apple A14 (iPhone 12 Pro) to the cross-device tables,
  plus an Avocado resolution sweep and a Python table for Apple devices.
- `README.md` / `README-ios.md`: macOS, iOS and Python-on-macOS build steps. The
  macOS pytest command was run as written (20 passed).
- `scripts/run_ios_benches.py`: new device runner, tested on image_fitting only;
  the full run used the scratch driver it was derived from.
- `run_benches.sh` / `.ps1`: viewer off for image_fitting and pbr_shading, as the
  README table assumes. The `.ps1` change was not run.
- `dressi_native_bench.py` / `dressi_torch_bench.py`: mean -> median, per the
  benchmarking rule.

## Method

M5 Pro on AC power: `run_benches.sh` three times (median), Avocado sweep three
times, Python benches three times, interleaved with C++ aa for the cross-check.
iPhone: viewer off, two runs (mean), 90 s rest between examples and 30 s in the
sweep. Raw records are in git-ignored `bench_results/`.

## Measurements

Median ms/iter:

| example | M5 Pro | A14 | quality (M5 Pro / A14) |
| --- | ---: | ---: | --- |
| image_fitting | 0.324 | 2.38 | max err 0.0005 |
| texture_optimization | 4.663 | 52.06 | 98.7% |
| silhouette (aa) | 2.049 | 32.55 | IoU 0.9544 / 0.9520 |
| silhouette (hardsoftras) | 6.418 | 28.88 | IoU 0.9507 / 0.9502 |
| pbr_material | 4.107 | 57.41 | 20.65 dB |
| pbr_envmap | 45.78 | 362.0 | 18.11 dB |
| pbr_shading, viewer off | 0.81 (about 1200 FPS) | 13.45 (74 FPS) | |
| pbr_shading, viewer on | 1.714 (583 FPS) | 11.67 (86 FPS) | |

C++ vs Python cross-check, M5 Pro, 256 / 512 / 1024: C++ aa 0.44 / 0.48 / 0.59;
native fused 0.42 / 0.46 / 0.54 in its fast mode, 0.61 / 0.69 / 0.87 in its slow
mode. Within noise in the same mode, closing the open item from the previous log.

## Correction: pbr_shading looked faster than NVIDIA

The first version of the tables showed the M5 Pro at 1177 FPS against 677 / 961
for the RTX cards. It was a condition mismatch, spotted by the user. The older
rows were recorded by `run_benches` with `--frames=60` only (log/202607131450),
which leaves the viewer on, while the Apple rows had `--view-interval=0`. Five
interleaved runs on the M5 Pro: viewer on 1.714 ms, off 0.812 ms. The tables now
use viewer-on rows, with the off figures in a footnote.

The Mac figure itself was checked and is real: frames render correctly, 1000
frames reproduce 0.81 ms, and GPU timestamps put about 1 ms in the one fused
194-function pass (inflated by the timestamps themselves). The viewer's effect
has no known mechanism and goes the other way on the iPhone, where viewer on is
faster: 11.67 vs 13.45 ms. The remaining lead over the RTX 3090 (583 vs 339 FPS)
is attributed to host and transfer cost, without an NVIDIA breakdown to prove it.

A14 viewer-on, one run, for comparison with the Android rows: image_fitting 2.55,
aa 32.6, hardsoftras 30.3, texture 62.2, pbr_material 55.2.

## Findings

- The first cross-check looked like a gap. Part of it was the native bench
  reporting a mean; the rest is a two-mode behaviour on sub-millisecond runs that
  C++ shows too. Cause unknown.
- iPhone figures moved with rests between examples: envmap 412 -> 362,
  texture 57.3 -> 52.1. Envmap still slows during its own run (running median 299
  at iter 200, 360 at the end), presumably thermal, which also clamps its
  `warmup_ms` to 0.
- iPhone Avocado HardSoftRas runs out of memory: K=3 at 4096 reports
  `VK_ERROR_OUT_OF_DEVICE_MEMORY`; K=3 at 2048 and K=1 at 4096 die with signal 9
  during the build, with no jetsam report, so their cause is presumed.
- A failed app run leaves the previous `bench.json` in the container, and the
  driver copied it, which gave repeated 86.917 values. Those were discarded, and
  `run_ios_benches.py` only copies after exit 0.
- Packing is identical on M5 Pro and A14.

## Open issues

- Two-mode timing on M5 Pro is unexplained.
- `pbr_shading` should be measured again on NVIDIA and Android with the viewer
  off; until then the table stays viewer-on.
- nvdiffrast and DRTK have no Apple figures: CUDA only.
- HardSoftRas at high resolution on iPhone would need a smaller memory footprint.
- Windows, Linux and Android remain unrebuilt from the previous log.
