# Desktop GPUs at full clocks: record and check the P-state

## Goal

The RTX 3090 report (PR #2 issuecomment-6772742409) showed that with no viewer
its driver keeps the GPU in a power-saving state, so `none` measured a
downclocked GPU on the GPU-bound rows. Make full performance a measurement
condition and make it checkable.

## What was done

- `run_benches.ps1` and `run_benches.sh` sample `nvidia-smi` every 200 ms
  during each run into `gpu_state.csv` next to `bench.json`, when `nvidia-smi`
  exists.
- `bench_summary.py` prints a "GPU state warning" under a table when a run's
  most common P-state among samples of the benchmarked GPU is not P0. Fewer than
  3 samples are not judged.
- docs/benchmarks.md: the full-performance rule under "Device state", the
  NVIDIA setting, the check in "Check a result", and the RTX 3090 status.
  CLAUDE.md summarizes it.

## Measurements

RTX PRO 6000, driver 597.30, pbr_envmap, two runs per condition:

| condition | ms/iter | GPU during the run |
| --- | --- | --- |
| `none` | 8.712, 8.599 | P0, 2347 MHz |
| `idle` | 8.623, 8.633 | P0, 2347 MHz |

RTX 3090, driver 610.88 (from the PR): `none` 31.09 ms at P3, 780 MHz;
`idle` 13.79 ms at P0, 1920 MHz.

Full run through `run_benches.py` with sampling, against the clean session at
the same build: all 21 runs at P0 in every sample, no warning. Every row is
within 1%. pbr_envmap reads 8.669-8.733 ms against 8.608-8.650, so the
ranges miss by 0.02 ms. Whether sampling or session drift causes that is not
separated.

## Decisions

- Keep `none` as the reference and require full clocks, rather than switching
  desktops to `idle`: `idle` relies on a driver heuristic about windows, and
  the PRO 6000 and the Linux Ada do not need it.
- Judge by P-state, not clock ratio: the PRO 6000 runs at 2347 MHz against a
  reported maximum of 3090 MHz, so a clock threshold would misfire.
- Match samples by GPU name, so llvmpipe on an NVIDIA host is not judged by
  the idle discrete GPU.

## Verification

- Synthetic samples: a mostly-P3 run warns, a P0 run with an idle first sample
  does not, a two-sample run and an llvmpipe record are not judged.
- The PowerShell runner (via `run_benches.py`) and the bash runner (Git Bash)
  both wrote `gpu_state.csv` and left no `nvidia-smi` running.

## Open issues

- Neither "Prefer maximum performance" nor PowerMizer has been tested. The RTX
  3090 re-run is the first test.
- AMD and Intel GPUs are not sampled. The Iris Xe run has no check.
- Iris Xe results have not arrived.
