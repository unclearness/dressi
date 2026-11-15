# bench_summary: warmup as the median across runs

## Goal

Review the macOS and iOS results (PR #2 issuecomment-6680329281, log
202611061715) and fix what stops them from entering the tables.

## What was done

- Checked the posted tables: all 14 rows are `viewer: none`, `runs` 3, commit
  `fa748c160375`, with quality and packing matching the older rows.
- `bench_summary.py` took warmup from the run behind the ms/iter median, so a
  cold start could reach the table (A14 hardsoftras 8985 ms against about
  1340 ms). Warmup and first build are now each the median across runs.
- Stated the rule in docs/benchmarks.md and CLAUDE.md.

## Decisions

- Median across runs rather than dropping run 1: it needs no rule for which run
  is cold, and a one-run row keeps its value.

## Verification

- Synthetic records with the A14 hardsoftras warmups give 1339 (891). A record
  without warmup fields still prints `-`.
- Old and new script on the 21 records in `bench_results/`: only the warmup
  column differs, on 3 of the 7 rows.

## Open issues

- A14 image_fitting (1.572, 2.188, 2.120 ms) is unexplained. The runner goes
  through every entry per pass, so run 1 was the first of the session and runs 2
  and 3 followed pbr_envmap (about 9 minutes) and a 90 s rest. Heat is a
  hypothesis only. A rerun of that entry alone is requested in the PR.
- docs/benchmarks.md and README tables wait for Windows, Adreno, and Mali.
