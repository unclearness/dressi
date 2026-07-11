# Intel iGPU portability of the examples

Goal: make the `examples/` run on this machine's Intel Iris Xe integrated GPU.
They had only ever been exercised on NVIDIA; the suspicion was that stage
packing bakes in NVIDIA-typical Vulkan limits and would emit shaders/pipelines
Intel cannot execute.

## Environment

- Sole Vulkan device: `Intel(R) Iris(R) Xe Graphics`, driver Intel Corp,
  apiVersion 1.4.323 (SDK 1.4.350). No discrete GPU, so `GetFirstPhysicalDevice`
  ([src/vk/context.cpp:9](../src/vk/context.cpp#L9)) correctly picks Intel.
- Key Intel limits vs NVIDIA-typical:
  - `maxPerStageDescriptorInputAttachments = 7` (NVIDIA >= 8) **← the blocker**
  - `maxImageDimension2D = maxFramebufferWidth = 16384` (NVIDIA 32768) **← blocker 2**
  - `maxColorAttachments = 8`, `maxPerStageDescriptorSampledImages = 200`,
    `maxPerStageDescriptorStorageImages = 16`, `maxPushConstantsSize = 256`,
    `maxComputeWorkGroupInvocations = 1024`.
- Format capability (measured with a standalone `vkGetPhysicalDeviceFormatProperties`
  probe): `R32_SFLOAT`, `R32G32_SFLOAT`, `R32G32B32A32_SFLOAT` **all** support
  `STORAGE_IMAGE` + `COLOR_ATTACHMENT` + linear sampling on this Intel part.
  The suspected RGBA32F-as-storage risk is a non-issue here; the blanket
  `eStorage | eColorAttachment | ...` usage in
  [CreateVarImage](../src/vk/executor.cpp#L16) is fine.

Most of the portability groundwork was already present: packing limits are
clamped from real `VkPhysicalDeviceLimits` in
[dressi_ad.cpp:121-134](../src/dressi_ad.cpp#L121-L134) (the `8/16/32` literals
are `std::min` caps, not NVIDIA hardcodes), codegen emits plain `#version 450`
with no extensions / subgroup / push-constant use, and `local_size_x=64` is well
within Intel limits. `geometryShader` (needed for `gl_PrimitiveID`) is supported.

## Blocker 1 — SumPixelWise fan-in > maxPerStageDescriptorInputAttachments (FIXED)

`silhouette_optimization data/bunny --technique={hardsoftras,aa}` aborted in the
packer:

```
No function is packable into an empty substage; candidates: SumPixelWise
  (src/pack/greedy_packer.cpp:486)
```

Root cause: the examples reduce per-view losses with a single n-ary
`F::SumPixelWise(view_losses)`. With the default `n_views = 8`, that node needs
8 same-pixel **input attachments** in one substage. Intel caps
`max_input_attachments = min(16, 7) = 7`, so the single function fit in no
substage and the packer (which cannot split one function) gave up. On NVIDIA the
limit is >= 8 so it fit. Confirmed by bisection: `--views=7` packed, `--views=8`
failed.

Fix ([src/core/f_ops.cpp](../src/core/f_ops.cpp), `SumPixelWise`): fold a wide
sum into a balanced tree of bounded fan-in (`kMaxArity = 4`, the Vulkan-guaranteed
`maxPerStageDescriptorInputAttachments` floor) — the exact pattern
`build_backward.cpp`'s `SumContribs` already uses for gradient accumulation. The
single-node builder is factored out as `SumPixelWiseFlat`.

Why this does not regress NVIDIA: the tree only gives the packer the *freedom* to
split. RSP fuses consecutive same-pixel elementwise ops, so on a device whose
input-attachment budget covers the full fan-in (NVIDIA, 8) the 4+4+2 tree
re-fuses into one substage with 8 input attachments — identical packing to the
old single node. Only where the budget is exceeded (Intel, 7) does it physically
split. Sum is associative and each term contributes gradient 1, so forward and
backward are unchanged.

Verified: CPU 87/87 and GPU 39/39 tests pass (both Naive and RSP). After the fix
`--technique=hardsoftras --views=8` converges on Intel.

## Example status on Intel after the fix

| example | status |
|---|---|
| texture_optimization (bunny) | works — 99.5% texel recovery |
| silhouette hardsoftras (bunny, 8 views) | works |
| silhouette aa (bunny, 8 views) | **fails — blocker 2** |
| silhouette aa (Avocado, 682f) | works |
| image_fitting | works |
| pbr_shading (DamagedHelmet) | works |
| pbr_material_optimization | works |
| pbr_envmap_optimization | works, slow (~156 ms/iter; EquirectSampleBwd
  transpose stages are ALU-heavy on Intel, e.g. one 64x1024 stage = 91 ms) |

## Blocker 2 — oversized 1D data image (FIXED via 2-D tiling)

`--technique=aa data/bunny` fails at image creation:

```
vkCreateImage: extent.width (30338) exceeds maximum image extent width 16384
  for R32G32B32A32_SFLOAT  (also > maxFramebufferWidth 16384)
```

`30338 = bunny.numFaces()`. The aa target pre-render
([main.cpp:340](../examples/silhouette_optimization/main.cpp#L340)) reads
`b_faces_f` (VEC3 `{n_faces,1}`) as a `texelFetch` image inside `F::AntiAlias`'s
vertex gather. hardsoftras avoids this — its target is a plain `F::Rasterize`
whose faces ride an index buffer, not a wide image. Any mesh with
> `maxImageDimension2D` (16384 on Intel; 32768 on NVIDIA) faces/vertices hits
this; typical meshes (DamagedHelmet 15452f, Avocado 682f) are under it.

Fix: store an oversized `{N,1}` table 2-D-tiled and fold the linear index in
codegen, keeping the untiled path byte-identical so NVIDIA never changes.

- `PhysImgSize` / `TileWidth` / `NeedsTiling`
  ([include/dressi/types.h](../include/dressi/types.h)): an oversized `{N,1}`
  image becomes `{W, ceil(N/W)}` with `W` the largest power of two `<= max_dim`.
  Power-of-two `W` makes the fold `(i & (W-1), i >> log2 W)` — matching the
  row-major tile — and, when `N` fits, `W == N` so it degrades exactly to
  `ivec2(i, 0)`.
- Allocation + guard
  ([CreateVarImage](../src/vk/executor.cpp#L16)): images allocate at the
  physical (tiled) size; a *computed* image this wide would also be a render
  target whose framebuffer/viewport would exceed the same limit, so it is
  rejected loudly (only static leaves — the per-face/-vertex tables — tile).
- Transfer ([transfer.cpp](../src/vk/transfer.cpp)): `SendImageToDevice` /
  `RecordImageUploads` upload the logical `{N,1}` rows into the physical tile's
  head (rest zero-padded); the copy extent follows `img->view_size`.
- Codegen ([glsl_codegen.cpp](../src/codegen/glsl_codegen.cpp)): `max_dim`
  threaded into `GenerateFragShader`/`GenerateCompShader`/`GenerateRasterShaders`
  (default `0` = no tiling, so golden tests and off-device callers are
  unchanged); `IdxCoord()` emits the fold, applied inside the two
  sampler-parameterized helpers (`dressi_face_cross`, `dressi_aa_pair`) which
  are the reachable tiled reads. A guard rejects a tiled table reaching any
  other index op (no example mesh is that large; the aa-target path via
  `dressi_aa_pair` is covered).
- Wired from [dressi_ad.cpp](../src/dressi_ad.cpp) and
  [executor.cpp](../src/vk/executor.cpp) with `ctx->limits.maxImageDimension2D`.

No-regression argument: with `max_dim=0` (tests) or with meshes under the limit
(NVIDIA at 32768; Intel's own small optimized meshes) `IdxCoord` returns the
historical `ivec2(i, 0)` verbatim, so the emitted GLSL is byte-identical — the
golden codegen tests ([test_codegen.cpp](../tests/test_codegen.cpp)) confirm it,
and every tiling branch (allocation, transfer, guard) is gated on
`NeedsTiling`, dead unless a table actually overflows. New CPU tests cover the
tile math and the codegen fold; `aa data/bunny` (30338-face target) now
converges on Intel to IoU 0.9458, matching the technique's expected result.

Verified after both fixes: CPU 89/89, GPU 39/39, vk-smoke 3/3, and all seven
examples run on the Intel Iris Xe.

## Open issues

- Only the AntiAlias / normal-consistency reads are tile-aware; optimizing a
  mesh with > maxImageDimension2D faces/vertices through the other index ops
  (RasterizeSoft, LookupFaces, the WIDE gathers) hits the codegen guard rather
  than working. Extend `IdxCoord` to those inline fetch sites when such a mesh
  is actually needed. `{V, max_deg}` adjacency tables (width `V`) are a separate
  tiling case not handled (they fail at image creation, loudly).
- COMP storage-image count is not checked against
  `maxPerStageDescriptorStorageImages` (16 on Intel); not hit by any example yet.
- pbr_envmap EquirectSampleBwd is very slow on Intel (~90 ms for a 64x1024
  transpose stage, ~156 ms/iter total); forward-correct but a perf wart.
