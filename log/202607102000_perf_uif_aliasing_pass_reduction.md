# Performance: uif_vars, zero-copy optimizer aliasing, pass reduction

Goal: work down the documented performance backlog in the order agreed
with the author: (1) `{1,1}` values as real uniforms (uif_vars), (2)
zero-copy optimizer aliasing, (3) intermediate image format / pass-cost
reduction. Benchmark task: Avocado (406v/682f) silhouette
self-reconstruction, 1 view, full iteration (fwd + bwd + in-graph Adam),
RTX PRO 6000.

## Instrumentation first: per-stage GPU timestamps

`GpuPlan` now carries a `vk::QueryPool`; when debug logging is on
(`SPDLOG_LEVEL=debug`), `BuildGpuPlan` writes a timestamp after the
frame-start copies, after every stage, and after the copy-backs, and
`ExecuteGpuPlan` reports per-stage averages every 100 frames with a
label (`F 256x256 ss=8 fn=270 i=68 s=10 o=59 [Compose,Mul,...]`).
`DRESSI_VK_DEBUG=1` additionally turns on the validation layers.

Baseline breakdown @256²/1view (HSR K=1, total 172-178 us GPU):
- 15 separate 256² FRAG render passes (backward chain) ~64 us — mostly
  per-pass fixed cost (~4 us each), split by `max_stage_attachments=16`
- 1×1 scalar stage (Adam powers + SumAll regularizer RMS) 27.9 us —
  SumAll's whole-image loop runs in ONE fragment thread (serial 406-texel
  loops × 3)
- end-of-frame optimizer copy-backs 13.3 us
- raster 16-18 us, FaceFetchBwd 11-14 us, vertex-grad gathers ~11 us

## 1. uif_vars ({1,1} leaves as real uniforms) — implemented, scoped by data

- `SubStage::uif_vars` (already in the IR, previously unused) is now
  populated by the greedy packer: a `{1,1}` SamePixel-broadcast input
  becomes a uniform **iff it is a leaf**; GPU-generated `{1,1}` scalars
  stay `texelFetch(0,0)` slt reads.
- Codegen emits one `vec4` UBO binding per uif var (`u_uifN`, bindings
  after slt); the executor allocates a 16-byte device-local UBO per
  variable (`plan.uif_bufs`, persists across rebuilds) and refreshes it
  with an image→buffer copy at frame start (leaves) or after the
  producing stage (computed values, kept for generality).
- **Why leaves only:** measured with GPU-generated scalars also routed to
  UBOs, the refresh copies + layout transitions cost ~+10 us/frame while
  the saved broadcast fetches are ~free (L1-cached) — a net regression
  (172→188-198 us). The paper's uniforms are CPU-side constants; that is
  exactly the leaf case.
- Test: `ExecutorRsp.UniformLeafUpdatePropagates` (sendImg update of a
  `{1,1}` leaf reaches the UBO on the next execStep). Existing `{1,1}`
  broadcast parity tests now exercise the UBO path.
- Effect on the benchmark graph: neutral (its dynamic scalars are all
  GPU-generated; its constants are inlined literals) — recorded honestly.

## 2. Zero-copy optimizer aliasing — copy-backs 13.3 → 0 us

The updated variable now writes **directly into its input leaf's
VkImage**. The Adam update substage reads pos/m/v same-pixel
(subpassLoad) and writes the update — so the shared image is bound as
BOTH input attachment and color attachment of the same subpass in
`eGeneral` layout with a subpass self-dependency. A fullscreen pass has
exactly one fragment per pixel, so no fragment ever reads another's
write (the legacy-legal Vulkan feedback pattern; validation-clean).

- Alias conditions checked per `[upd, inp]` pair in `BuildGpuPlan`
  (writer substage consumes inp same-pixel only; nothing else in the
  writer stage touches either; no later stage reads inp). Pairs that
  fail keep the end-of-frame copy-back — behavior, not API, changed.
- WAR hazard: an execution barrier (fragment/transfer reads →
  color-attachment write) precedes any stage writing aliased images;
  the post-stage barrier gains transfer-read visibility for next
  frame's vertex-buffer/UBO refresh copies.
- `recvImg`/`sendImg` semantics unchanged (same image either way).

## 3. Pass-cost reduction

- **Stage merging:** `max_stage_attachments` 16 → 64 (a render pass has
  no hard Vulkan cap on total attachments) and `max_input_attachments`
  8 → 16. The 15-pass 256² backward chain collapses into ONE render
  pass of 8 chained subpasses (ByRegion dependencies): ~64 → ~32 us.
  26 stages → 11 stages overall.
- **SumAll two-level reduction:** `F::SumAll` now inserts a
  `__sum_partial__` `{64,1}` strided partial-sum pass when the input
  exceeds 256 pixels; the final `__sum_all__` loop visits 64 texels
  instead of the whole image. The 1×1 scalar stage: 30.4 → 3.9 us
  (+4.7 us partial stage shared by all three SumAlls).
- **Formats:** not changed. FLOAT/VEC2 images are already R32F/RG32F;
  fp16 would break the finite-difference/CPU-parity tolerances and
  gradient quality; the remaining known waste is the VEC3→RGBA32F
  padding (color-attachment support for RGB32F is not guaranteed).

## Results (median execStep, viewer disabled via new --no-view)

GPU-time breakdown @256² went 172-178 → 97.5 us. End-to-end:

| resolution | HSR K=1 before → after | K=3 before → after | AA before → after |
|---|---|---|---|
| 256²  | 0.20 → **0.16** | 0.39 → **0.27** | 0.49 → **0.48** |
| 512²  | 0.28 → **0.21** | 0.65 → **0.43** | 0.52 → **0.49** |
| 1024² | 0.58 → **0.41** | 1.38 → **1.03** | 0.54 → **0.51** |
| 2048² | 1.54 → **1.22** | 4.49 → **3.47** | 0.60 → **0.57** |
| 4096² | ~5.5 → **4.37** | — | 0.81 → **0.78** |

- HSR K=1 now also beats nvdiffrast CUDA at 2048² (1.22 vs 1.28) and
  narrows the 4096² pixel-bound gap (4.37 vs 3.28-3.47).
- Paper comparison @256²: 0.304 ms (RTX 2080) vs 0.16 ms.
- Quality: identical to pre-session (defaults, 300 iters: HSR IoU 0.957
  vs 0.957 baseline, AA 0.940 vs 0.936 — verified by stashing and
  re-running the pre-session build). All 98 tests pass. Numeric traces
  differ slightly run-to-run because shader fusion changes FMA
  contraction patterns; parity tolerances hold.

## Remaining known levers (updated)

- Per-pass fixed cost still dominates ≤512² (11 passes ≈ 45 us of the
  97.5); COMP-shader substages and rematerialization (recompute cheap
  values instead of crossing subpass boundaries through attachments)
  are the paper-side items left.
- The 256² backward mega-stage (27 us) is attachment-bandwidth-bound
  (59 RGBA32F outputs across 8 subpasses); fewer materialized
  intermediates would cut it further.
- HSR 4096² remains pixel-bound vs nvdiffrast's hand-tuned kernels
  (4.37 vs 3.3).
