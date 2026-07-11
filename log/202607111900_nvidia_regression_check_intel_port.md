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

## Follow-up: why HSR (and only HSR) re-packs differently

Investigated with temporary instrumentation in `greedy_packer.cpp` (a debug
dump of every final substage's function-name sequence + a trace of
SumPixelWise placement attempts), run on both builds and diffed. The
instrumentation was reverted after the investigation.

1. **The unordered-container hypothesis is ruled out.** The only
   `std::unordered_map` in packing (`pending_consumers`) is find/count-only;
   every unordered_set in build_backward/traverse/reactive is a pure
   membership filter. Nothing order-dependent ever iterates an unordered
   container — iterated maps are `std::map<Variable,...>` under the
   node-id `std::less`. Empirically: 3 HSR runs are bit-identical, so the
   packing is deterministic; it changed *between commits*, not between runs.

2. **It is not an attachment-budget failure at the loss sum.** On this GPU
   `max_input_attachments = min(16, dev) = 16`; the traces show the 4+4+2
   tree nodes joining substages with room to spare, and even the old flat
   8-input sum was well under 16.

3. **Actual mechanism: candidate-unlock order in the back-to-front greedy
   packer.** With the flat 8-ary sum, packing it made all 8 per-view loss
   branches become candidates *simultaneously*; with the tree, S3 unlocks
   {S1,S2}, then S1/S2 unlock their 4 views in two groups at different
   times. The `CountEdges`-preference + id tie-break then interleaves the
   8 HSR view-branches differently, and the greedy fusion cuts different
   substage boundaries throughout the forward/backward view blocks (e.g.
   base per-view groups of 12/19/22/142 funcs become 13/27/14/65/20/20/111
   regroupings; net +1 substage). Boundaries don't change per-func math
   (fp32 attachments), but the driver's shader compiler contracts
   mul+add→FMA *within* one shader, so different fusion groupings shift
   low-bit rounding; HSR's Step/Sigmoid thresholds amplify that into the
   observed small trajectory change. In AA the diff is exactly ONE
   substage (funcs 51→53: the 3 tree nodes fused where the flat node used
   to sit; every other boundary byte-identical), and the gradient of a sum
   is exactly 1 regardless of grouping — hence AA's bit-identical run.

Conclusion: deterministic, packing-order butterfly effect confined to the
HSR graph; quality within historical session range, wall time unchanged.
If exact NVIDIA-invariance is ever wanted, fold wide sums at *pack* time
(device limits in hand) instead of at graph-build time — invasive, since
the packer currently cannot split a single function; not worth it today.
