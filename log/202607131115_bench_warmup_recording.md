# Record warmup/build cost in bench.json (cross-device)

Branch: `cross-device-bench` (PR unclearness/dressi#1)

## Goal

The `bench.json` records (added by this PR) only carried the median
steady-state ms/iter with the warmup excluded. Requested: also always
record, cross-device, how long the initial warmup takes (stage packing,
GLSL compile, Vulkan build). Rough is fine.

## What was done

- Added `WarmupMs(warmup_samples, steady_median)` to
  `examples/common/bench.{h,cpp}`: one-time build/rebuild overhead =
  `sum(warmup_samples) - steady_median * N`, clamped to >= 0. Wall-clock,
  rough by design. Captures the full pack + GLSL-compile + Vulkan-build
  ladder AND the reactive-cache rebuilds at iter ~2 and ~8 (all inside the
  20-iter warmup window).
- All six examples now keep the excluded warmup-phase per-iter samples (an
  `else` branch on the existing `iter >= warmup` gate — same measurement
  point as the steady samples, execStep only, readback excluded) and write
  two new keys:
  - `warmup_ms`  — total one-time overhead
  - `first_build_ms` — iter 0 alone (the single largest build)
  Each example also logs the pair.
- `image_fitting/run.cpp` was the only example missing
  `spdlog::cfg::load_env_levels()`, so `SPDLOG_LEVEL=debug` had no effect
  there (no per-phase execStep breakdown). Added the include + call for
  parity with the other five.
- `scripts/bench_summary.py`: new `warmup ms (first)` column (via
  `warmup_of`); "-" for older records without the field. Docstring updated.
- CLAUDE.md "Benchmarking" section documents the new fields and points at
  `SPDLOG_LEVEL=debug` for the per-phase breakdown (substage-pack /
  vulkan-build / upload), which is the only source of the phase split (the
  `PhaseTimer` in `dressi_ad.cpp` is debug-only and not exposed via API).

## Measurements (RTX 6000 Ada / Xeon w7-3465X / Ubuntu 24.04.4)

| example | median ms/iter | warmup_ms | first_build_ms |
| --- | ---: | ---: | ---: |
| image_fitting | 0.042 | 258 | 254 |
| texture_optimization | 0.880 | 389 | 370 |
| pbr_shading | 1.06 | 561 | 550 |
| silhouette aa | 1.24 | 610 | 514 |
| silhouette hardsoftras | 3.09 | 1007 | 727 |
| pbr_material | 3.74 | 1180 | 909 |
| pbr_envmap | 8.36 | 896 | 778 |

Quality metrics unchanged from before this change (texopt 99.4%,
silhouette IoU 0.9562/0.9364, etc.).

## Key decisions / notes

- Wall-clock `first_build_ms` (~370 ms for texopt) runs ~2x the
  `SPDLOG_LEVEL=debug` PhaseTimer total (~185 ms) because the timer marks
  only object creation, not the driver-side shader compile / first-submit /
  first-run GPU-context init that the wall clock sees. The wall number is
  the honest end-to-end warmup the user pays; the debug breakdown stays the
  way to attribute it to phases. Documented as rough.
- Kept `warmup_ms` separate from `first_build_ms` so the reactive-rebuild
  tail (iter ~2 + ~8) is visible: e.g. silhouette-hardsoftras 727 -> 1007
  is a ~280 ms iter-8 full rebuild.
- No engine changes; `bench.{h,cpp}` links only into examples, so ctest is
  unaffected.

## Also in this session: `--view-interval` for the live viewers

Separate small request: the live viewers refreshed on a fixed cadence
(texture every 10 iters, silhouette every 5, pbr_material/pbr_envmap every
20, pbr_shading every frame). Made it a CLI option `--view-interval=N`
(iters/frames; `0` = never refresh) on all five viewer examples, defaulting
to **1** (refresh every iteration). image_fitting has no viewer, so it is
untouched.

- Guarded every use with `view_interval > 0 && iter % view_interval == 0`
  so `0` disables refresh without a modulo-by-zero.
- **Does not affect the benchmark**: the timed region wraps `execStep`
  only; the viewer refresh (recvImg + `viewer->update`) is outside it (its
  own `view_ms`, excluded), and headless runs skip the viewer block
  entirely (`viewer_open == false`). Verified: texture median 0.886 and
  silhouette hardsoftras 3.2 ms/iter unchanged with the new default.

## Open issues

- Per-phase breakdown is still debug-log only. A `DressiAD` accessor that
  returns the last rebuild's PhaseTimer entries would let bench.json record
  `build_substage_pack_ms` / `build_vulkan_ms` cross-device, but that's an
  engine API change beyond the "rough" scope here.
