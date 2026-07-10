# Python silhouette steady-state benchmark

## Goal

Establish a fair, repeatable baseline for the CUDA-tensor path of
`examples/python/silhouette_optimization.py` before further eager-path
optimization.

## Changes

- Exclude 20 warmup iterations so Dressi graph construction and cache
  initialization do not inflate the reported optimization time.
- Time synchronized 10-iteration blocks and report the median block-average
  throughput. Per-iteration CUDA synchronization was deliberately avoided
  because it changes the workload being compared.
- Move progress loss readbacks and printing between timed blocks.
- Document the CUDA block-timing rule in `CLAUDE.md` and refresh the README
  multi-view figures.

## Measurements

RTX PRO 6000, 8 views, 128x128, 300 iterations, CUDA tensors. Each Python
backend was run three times; the table reports the median run.

| path | median ms/iter | observed IoU range |
|---|---:|---:|
| dressi.torch AA | 8.41 | 0.9717-0.9738 |
| nvdiffrast CUDA AA | 3.09 | 0.9829-0.9858 |
| dressi.torch HardSoftRas | 20.73 | 0.9612-0.9620 |
| Dressi C++ AA | 1.237 | 0.9364 |

The C++ graph is not numerically identical to the Python eager program, but
its result bounds the engine-side cost: most of the 7.2 ms Python/C++ gap is
in the three synchronized eager operation boundaries per iteration.

## Verification

- `uv run python -m py_compile examples/python/silhouette_optimization.py`
- `uv run pytest tests/python -q`: 20 passed
- C++ AA headless run completed at 1.237 ms/iter steady-state
- `git diff --check`

## Next

Combine each eager operation's upload, render execution, and readback into a
single queue submission and fence wait, then re-run the same benchmark.

## Follow-up: combined upload, render, and download

Completed the follow-up on top of `2955526` and `f561a33`:

- Split stacked readback into prepare, command-recording, and resolve phases,
  preserving host-cached staging and VEC3 unpacking.
- Submitted optional upload commands, the cached render plan, and download
  commands as one ordered command-buffer array with one fence wait.
- Added `DressiAD::execStepWithSendsAndRecvImgsStacked`, its nanobind binding,
  and `Engine.run_and_read_{outputs,grads}_stacked()`; replaced adjacent
  run/read pairs across the eager ops. A gradient engine falls back once
  before its gradient Variables have been materialized by the first build.

Same-session A/B on the RTX PRO 6000 (same benchmark methodology as above):

| technique | combined runs (ms) | combined median | legacy runs (ms) | legacy median | effect |
|---|---|---:|---|---:|---:|
| AA | 7.98 / 7.92 / 7.98 | 7.98 | 8.27 / 8.21 / 8.27 | 8.27 | -3.5% |
| HardSoftRas | 21.58 / 21.60 / 21.69 | 21.60 | 20.89 / 21.67 / 21.98 | 21.67 | -0.3% (noise) |

Combined-path IoU stayed at 0.9717-0.9727 (AA) and 0.9605-0.9619
(HardSoftRas). The prior 20.73 ms HardSoftRas number initially appeared to
show a regression, but the same-session legacy control identified normal
run/system variance. The shared C++ AA path was unchanged at 1.211 ms/iter
(prior 1.237), IoU 0.9364.

Verification: Release build completed; CTest 105/105 and Python pytest 20/20
passed; `py_compile` and `git diff --check` passed. The result is a repeatable
3.5% AA win and a neutral result for the more compute/host-heavy HardSoftRas
path. Python/autograd/tensor plumbing remains the next material bottleneck.
