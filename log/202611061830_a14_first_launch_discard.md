# A14 image_fitting spread: discard the first pass after an install

## Goal

Explain the A14 image_fitting spread left open in log 202611061743, and settle
which figure the tables use. Results: PR #2 issuecomment-6681425704.

## What was done

- Re-summarized M5 Pro and A14 with the warmup median from `30b23fb`. Recomputing
  the A14 values from the posted per-run records matches the tables. ms/iter did
  not change on either device.
- Ran A14 image_fitting alone, after a 10-minute idle, on the same install.
- Added the discard rule to docs/benchmarks.md, CLAUDE.md, README-ios.md, and
  README-android.md, with the commands in the Android and iOS steps.
- Recorded in the docs' Current status that the Apple figures exist but are not
  in the tables yet.

## Measurements

A14 image_fitting, ms/iter:

| session | run 1 | run 2 | run 3 | median |
| --- | ---: | ---: | ---: | ---: |
| main pass | 1.572 | 2.188 | 2.120 | 2.120 |
| after 10-minute idle | 2.130 | 2.162 | 2.167 | 2.162 |

## Findings

- The heat hypothesis from log 202611061743 was wrong: a rested phone gave about
  2.1 ms on all three runs, so neither heat nor running after pbr_envmap explains
  1.572 ms.
- The 1.572 ms run was the first launch after the reinstall (warmup 1420 ms
  against about 99 ms). The link is an association. No test isolates it.
- No other A14 row shows it. image_fitting's run 1 is 26-28% faster than its
  later runs in that pass, while texture_optimization and pbr_envmap_optimization
  differ by 4.2% and 3.3% across their three runs.

## Decisions

- The A14 image_fitting figure for the tables is 2.162 ms (2.130-2.167): same
  commit and condition, three runs in one session, no first launch in it.
- The other A14 rows keep all three runs, since their first pass is not an
  outlier. Re-measuring them was not worth the time.
- Mobile runs now start with one discarded pass after an install, using the
  existing `--reps 1 -o out/discard`, so no runner changed. It stays outside
  `bench_results` because `bench_summary.py` would count it as a fourth run.

## Open issues

- Why a first launch would run faster is unknown, and Android has not been
  checked for the same effect.
- docs/benchmarks.md and README tables wait for Windows, Adreno, and Mali.
- `--avocado` was not run, so the resolution sweep is still the old one.
