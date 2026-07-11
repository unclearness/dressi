# NVIDIA regression check for the Intel-iGPU portability commits

Goal: verify that `e3edb65` (Support Ubuntu) and `6ddce51` (Run the examples
on Intel integrated GPUs) do not regress the NVIDIA path. The Intel work was
verified only on the Iris Xe machine; this session re-validates on the
RTX PRO 6000 Blackwell Max-Q (driver 596.72), the machine all README
numbers come from.

## Method

- Full Release rebuild at `6ddce51` + entire ctest suite (GPU labels included).
- All seven examples run headless where possible.
- A/B against a `git worktree` build of the pre-change commit `197bd00`
  (submodule rolled back to `110dab3`), same binaries run back-to-back from
  the same working directory, same data. Stage counts compared with
  `SPDLOG_LEVEL=debug`.

## Results

Tests: **132/132 pass** (89 cpu + 43 gpu incl. python_bindings).

A/B (base `197bd00` → current `6ddce51`), median steady-state / final metric:

| example | base | current | verdict |
|---|---|---|---|
| silhouette aa bunny (8 views, 128²) | 1.218 ms, IoU 0.9364 | 1.218–1.220 ms, IoU 0.9364 | loss bit-identical (0.10412780 @299) |
| silhouette hsr bunny | 2.206 ms, IoU 0.9587 | 2.199 ms, IoU 0.9562 | see finding below |
| silhouette aa Avocado 512² single-view | 0.226 ms, IoU 0.6141, exit 1 | 0.229 ms, identical loss | bit-identical; exit 1 is the example's own convergence gate and pre-exists (single-view is a timing config, README baseline ~0.26 ms holds) |
| texture_optimization bunny (6 views) | 99.4 %, loss 0.00000842 | identical | bit-identical |
| pbr_material_optimization (6 views) | PSNR 21.22 dB, 4.12 ms | PSNR 21.22 dB, 3.89–4.10 ms | bit-identical numerics |
| pbr_envmap_optimization (4 views) | PSNR 16.40 dB, 8.85–8.96 ms | PSNR 16.40 dB, 8.90–9.00 ms | bit-identical numerics, timing overlap = noise |
| image_fitting | — | max err 0.0000 | OK |
| pbr_shading | — | 12 stages warmup, 1.379 ms/frame (725 FPS) | matches documented behavior |

## The one real (minor) NVIDIA-side change: HSR packing

`6ddce51` claims "NVIDIA packing is unchanged" because RSP re-fuses the new
bounded-fan-in SumPixelWise tree. Measured:

- aa bunny: 54 stages / 55 substages both builds (funcs 800 → 802, the two
  extra tree nodes) — fully re-fused, trajectory bit-identical. Claim holds.
- **hsr bunny: 65/132 → 66/133** — one tree node did NOT re-fuse, leaving
  +1 stage / +1 substage. The changed summation grouping shifts float
  rounding, and HSR's discrete branches (Greater masks, stochastic
  FaceFetch) amplify it into a slightly different — but fully deterministic
  (3 runs identical) — trajectory: loss 0.0861 → 0.0992 (still 21.6x
  reduction), IoU 0.9587 → 0.9562, flips 8 → 11. Wall time unchanged
  (2.206 → 2.199 ms). Within the historical HSR session range (0.957–0.9587
  in earlier logs); judged not a meaningful regression, but the commit
  message's "unchanged" is not exact for HSR.

## Open issues

- Optional: investigate why RSP leaves the split in the HSR graph (likely an
  attachment/limit budget already consumed in the target substage) if the
  +1 substage or the IoU wiggle ever matters.
