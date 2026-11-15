# macOS and iOS re-measurement under the defined conditions

## Goal

Measure the M5 Pro and A14 again at `fa748c1`, following the steps in PR #2
(issuecomment-6679004798) and "Measurement conditions" in docs/benchmarks.md.
Results: PR #2 issuecomment-6680329281.

## What was done

- macOS: build, `ctest` 133/133, `run_benches.py` (3 runs, viewer `none`).
- iOS: rebuild, DressiTests 132/132 on the iPhone 12 Pro, `run_ios_benches.py`
  (3 runs, viewer `none`). 42 of 42 runs exited 0. No repository changes.

## Measurements

Median of three run medians, ms/iter (min-max):

| example | M5 Pro | A14 |
| --- | ---: | ---: |
| image_fitting | 0.285 (0.277-0.286) | 2.120 (1.572-2.188) |
| texture_optimization | 4.658 (4.653-4.660) | 52.582 (51.524-53.669) |
| silhouette aa | 2.053 (2.045-2.096) | 31.659 (31.353-31.758) |
| silhouette hardsoftras | 5.903 (5.795-5.908) | 29.537 (29.454-30.008) |
| pbr_material | 4.620 (4.295-4.646) | 56.793 (53.686-56.872) |
| pbr_envmap | 45.791 (45.784-45.802) | 366.912 (358.063-370.000) |
| pbr_shading | 0.735 (0.733-0.746), 1360 FPS | 13.287 (13.040-13.469), 75 FPS |

## Findings

- A free developer profile allows three apps, and the phone held two others.
  DressiExamples had to be uninstalled to install DressiTests, then swapped back.
- Reinstalling makes run 1 of each example a cold start: hardsoftras warmup
  8985 ms against about 1340 ms afterwards. Presumably an empty shader cache.
  `bench_summary.py` shows the warmup of the median run, so a cold value can
  reach the table. Steady-state medians are not visibly affected.

## Open issues

- A14 image_fitting spread (1.572 against 2.12-2.19 ms) is unexplained.
- "Quit other apps" on the Mac was not verified; the check timed out.
- The Avocado sweep (`--avocado`) was not run, and the tables in
  docs/benchmarks.md and README.md were not updated.
