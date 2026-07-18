# Algorithm & architecture

How dressi turns a graph of image operations into packed Vulkan render passes,
and how the differentiable-rendering pieces are built on top. See the
[README](../README.md) for the overview and [benchmarks.md](benchmarks.md) for
performance.

## The execStep ladder

Each `DressiAD::execStep()` walks the paper's InitStatus ladder:

```
build backward graph → wire optimizer → traverse → pack substages →
pack stages → build Vulkan objects → submit
```

Each step reruns only when its inputs changed (a **reactive cache**), so a
steady optimization loop re-executes pre-recorded command buffers with nothing
crossing PCIe. The transfer model is the paper's: upload leaves once, download
results at the end.

## Milestone 1 — Dressi-AD core + Reactive Shader Packing

- **Graph core.** `Variable` / `Function` handles over a shared node graph
  (`shared_ptr` bodies, weak back-edges; creation id doubles as topological
  order). Type/size inference with `{1,1}` uniform broadcasting.
- **Operators (`namespace F`).** Each op carries three things: a forward GLSL
  snippet, a backward generator, and a CPU kernel. The op set covers
  arithmetic, the PBR math set (trig / exp / log / pow / min / max / clamp /
  mix / step / smoothstep), vector ops (dot / cross / normalize / reflect / …),
  matrix ops, and spatial reductions (`Sum` / `Mean` via a log-step 2×2 reduce
  chain). The CPU evaluator is the test oracle — every op is verified by
  forward parity and central finite-difference gradient checks.
- **`BuildBackward`.** Reverse-mode gradient graph with gradient-shape reduction
  for broadcast inputs. Losses stay images: gradient 1 is seeded per pixel
  (= derivative of the implicit sum), so no forced reduction to a scalar.
- **Packing.** Two modes: **Naive** (1 function = 1 pass, the correctness
  baseline) and **RSP** (Reactive Shader Packing): greedy back-to-front fusion
  of substages under each device's Vulkan limits, then `RematerializeSubStages`
  clones cheap producer chains into consumers and drops dead outputs. Long
  same-pixel chains collapse into one render pass of chained subpasses.
- **Reactive cache.** Dirty tracking, FAST/FULL rebuild thresholds, pruning of
  clean cached branches, automatic re-expansion when inputs change.
- **Zero-copy optimizer aliasing.** The optimizer writes updated values directly
  into the input leaf's image (shared image bound as input + color attachment in
  `eGeneral` with a subpass self-dependency — legal because a fullscreen pass
  has one fragment per pixel). GPU-resident optimizer *state* uses
  `DressiAD::addUpdate(state_leaf, new_state)` (in-graph Adam m/v/t).
- **`DressiAD` driver.** `setLossVar` / `setOptimizer` / `sendImg` / `recvImg` /
  `execStep`.

## Milestone 2 — deferred rasterization + texture optimization

- **`F::Rasterize(clip_pos, attrib, faces, screen_size)`.** Depth-tested indexed
  rasterization of a vertex attribute into a screen-space G-buffer channel
  (deferred shading, per the paper). Non-differentiable w.r.t. geometry (that is
  Milestone 3). Vertex data is uploaded into vertex/index buffers; computed clip
  positions are also legal raster inputs (the executor copies the producer image
  into a device-local vertex buffer each frame).
- **`F::Texture(tex, uv, inv_uv)`.** Nearest sampling, differentiable w.r.t. the
  texture via the paper's **inverse-UV lookup**: the mesh is rasterized in UV
  space with screen positions as the attribute, giving a per-texel screen
  coordinate to gather the upstream gradient from. Includes an occlusion guard
  and on-the-fly dilation for chart-boundary texels (the dilation was essential
  to reach loss 0).
- **Sobol-jittered sampling.** The paper's quasi-random countermeasure against
  texels that never receive updates: each iteration offsets the sampling UVs by
  a 2D Sobol point and renders the target in-graph with the GT texture under the
  same jitter, so fresh texels are constrained every step.
- **`examples/texture_optimization`.** Multi-view unlit rendering of `data/bunny`.
  From a black texture, the summed per-view MSE densely recovers the atlas
  (~640k texels, zero-jitter rendered loss ~5e-6).

## Milestone 3 — vertex-position gradients (HardSoftRas + screen-space AA)

Two selectable techniques give gradients that flow through in-graph projections
back to a 3D position leaf. Static topology rides in adjacency textures
(`VertexFacesTex` / `VertexNeighborsTex` / `FaceNeighborsTex`); the screen→vertex
backwards are per-vertex gathers bounded by adjacency bounding boxes (no atomics,
no compute).

- **`F::RasterizeSoft` — HardSoftRas (Dressi Alg. 2, `--peels=K`).** GPU-enlarged
  triangles (centroid scaling via `F::SoftClip`) supply a face-ID buffer with the
  paper's Eq. 3 depth shift in `gl_FragDepth`. The pixel-to-edge distance is built
  from ordinary image-space ops over `F::LookupFaces` / `F::FaceFetch`'d corner
  positions, so the backward is generated entirely by the AD ("no special backward
  declarations", per the paper). Gradients cross the indexed reads with the
  inverse-UV philosophy — stochastic jittered edge samples (Alg. 1; Vulkan lacks
  float atomics), Wang-hash jitter that is bit-identical between GLSL and the CPU
  oracle. Depth peeling `--peels=K` blends Eq. 6 layers (K=3 matches the exact
  reference).
- **`F::AntiAlias` + `F::RasterizeFaceId` — Dr.Hair (Eq. 3-5).** Blends across
  triangle-ID boundaries (`gl_PrimitiveID`) by the pixel-to-silhouette-edge
  distance `r = min(|e_s|/L, 1)`; differentiable w.r.t. both the image (an exact
  linear transpose) and the vertex positions (per-vertex gather, exact bbox scan
  or stochastic edge samples via `--samples`). Works on general triangle meshes.
- **WIDE gathers.** The per-vertex gather backwards (`GatherDistGrad`,
  `AntiAliasBwdVtx`) default to WIDE: `{V, max_deg}` partials (one incident face
  per thread) reduced by a `__col_sum__` pass — a `{V,1}` target runs only V
  threads and is brutally latency-bound. This is the key lever that also sped up
  the Python paths.
- **`examples/silhouette_optimization`.** Deforms an icosphere into the bunny (or
  self-reconstructs a glTF mesh via `--mesh=`) from multi-view silhouettes. The
  whole iteration is GPU-resident: in-graph projections, soft geometry,
  uniform-Laplacian / normal-consistency regularizers (`F::VertexNeighborMean`,
  `F::NormalConsistencyFaceTerm` / `VertexGrad`), and Adam with GPU-resident
  state — zero per-iteration CPU traffic. Mask IoU ~0.98 for both techniques.

## Milestone 4 — PyTorch bindings (`dressi.torch`)

An nvdiffrast-compatible eager API (`dressi._C` nanobind module + a
`dressi.torch` wrapper) backed by the Vulkan engine — no CUDA required.

Define-and-run ⇄ eager is reconciled by a **shape-signature-keyed engine
cache**: the first call with a new (resolution, mesh size, params) signature
compiles a `DressiAD` graph (packing + GLSL + Vulkan objects); subsequent calls
only upload changed leaf values, execute the cached command buffer, and download
results. External gradients enter through a surrogate loss `output * grad_seed`
(BuildBackward seeds 1 everywhere, so leaf gradients equal the exact VJP for the
uploaded `grad_output`).

- **Batching.** N-item calls unroll into ONE engine (one execStep per forward and
  one per backward for a whole minibatch); uploads/downloads ride single staging
  submits (`sendImgs` / `recvImgs`, same-shape batches stacked into one buffer);
  forward runs a gradient-free engine (backward stages never execute in forward).
- **Native fused path.** `scripts/dressi_native_bench.py` builds the whole
  training step — transform, raster, AA, MSE, in-graph Adam — as one dressi graph
  and matches the C++ example from Python (zero per-iteration PCIe traffic).
- **Transfer-layer lessons (9× eager speedup).** Readback staging must be
  HostCached (memcpy from write-combined memory runs at ~100 MB/s); staging
  buffers and one-shot command buffers are persistent per context; channel-
  identical images skip the re-stride pass.

Gotchas: `setRequiresGradRecursively` propagates DOWNSTREAM — call it AFTER
building the graph; an identity optimizer does NOT work for grad readback
(self-`CopyImage` UB — use `setGradOutputsEnabled` / `inputGrads`).

## Milestone 5 — physically based shading / IBL

A split-sum PBS layer (Cook-Torrance + image-based lighting in Dressi notation),
everything **equirectangular 2D** — no cubemaps, no hardware mips, no linear
samplers. All bilinear filtering is manual 4-tap `texelFetch` inside special ops
(`__equirect_sample__` u-wraps at the seam; hardware `texture()` bilinear would
break GPU↔CPU parity and the clamp sampler would break the seam), with CPU/GLSL
twin helpers in `core/ibl_math.h`. One shared forward graph drives three
examples:

- **`pbr_shading`.** Real-time DamagedHelmet viewer. Forward-only; the IBL
  precompute is pruned by the reactive cache (12 stages warmup → 5 steady).
- **`pbr_material_optimization`.** Albedo recovery via `F::TextureBilinear`'s
  tent-weighted inverse-UV gather (accumulates every contributing pixel, so
  stable steps are ~Σw² smaller than the nearest gather's).
- **`pbr_envmap_optimization`.** HDR environment recovery through the
  object's diffuse shading and (with the differentiable prefilter) specular
  reflections — the loss is masked to the helmet's coverage by default
  (`--fg-only=0` adds the direct background observation; it buys only
  ~0.15 dB because the metallic helmet already constrains almost the whole
  sphere). `F::EquirectSample` / `F::IrradianceConv` / `F::PrefilterConv` are
  deterministic linear maps with **exact transposes** (no atomics); the equirect
  transpose is WIDE-split into `{map_w, map_h·K}` band partials + `__tile_sum__`.
  A `--env-reg` smoothness prior resolves the underdetermined-inverse speckle
  (the `Max(env,0)` clamp otherwise rectifies null-space oscillation into black
  specks); env PSNR 16.4 → 18.3 dB (background-inclusive numbers).

## Known deviations from the paper (intentional)

- Appendix typos fixed: `Sin` backward, `BuildBackward` null check.
- GPU-generated `{1,1}` scalars stay `texelFetch(0,0)` (a mid-frame UBO refresh
  measured slower than the broadcast fetch); only `{1,1}` *leaves* ride real
  vec4 UBOs.
- COMP substages execute as compute dispatches, but by measurement only
  `GatherDistGrad` defaults to COMP — for the hot band-pass gathers, graphics↔
  compute switches plus lost elementwise fusion cost more than the render passes
  they remove.
- HardSoftRas triangle enlargement is centroid scaling (no geometry shader, no
  obtuse bbox split); the AA technique picks the boundary edge by owner
  preference rather than an exact closest-depth test (gradients, not forward
  continuity, are the deliverable).
- glTF `LoadGltfScene` does NOT flip V (glTF is top-left origin); the older
  `LoadGltfMesh` keeps its 1-v flip for compatibility.

## Not yet implemented

Texture backward w.r.t. UV, near-plane clipping, int/matrix images on GPU,
mipmaps / hardware bilinear (would break GPU↔CPU parity), zero-copy CUDA interop
for the eager Python path.

## Source layout

```
include/dressi/   public API (Appendix A style)
src/core/         graph nodes, F:: ops, BuildBackward, CPU evaluator, IBL math
src/codegen/      substage -> GLSL fragment/compute shader generation
src/pack/         trivial & greedy (RSP) packers, reactive pruning
src/vk/           headless Vulkan context, executor, CPU<->GPU transfer
src/python/       nanobind module (dressi._C) + dressi.torch eager API
```
