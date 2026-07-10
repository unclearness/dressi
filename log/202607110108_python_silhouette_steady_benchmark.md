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
