# PyTorch bindings (dressi.torch): nvdiffrast-compatible API on the Vulkan engine

## Goal

Provide an nvdiffrast-drop-in PyTorch API (`import dressi.torch as dr`)
backed by the Dressi engine: `rasterize` (non-differentiable) /
`interpolate` / `texture` / `antialias`, plus `rasterize_soft`
(HardSoftRas) as an extension — vendor-neutral (any Vulkan GPU),
permissively licensed, with performance approaching the C++ examples
(user requirement mid-session: 「C++と同等程度の速度が出ることが目標」).
v1 decisions (user): drop-in nvdiffrast signatures, CPU-tensor transfer,
HardSoftRas as an extension function. Python package name: `dressi`
(user request; the project keeps the name opendressi).

## The define-and-run ⇄ eager reconciliation

- Every torch op is a `torch.autograd.Function` over a **shape-signature-
  keyed engine cache** (`RasterizeVulkanContext._engines`): cache miss
  builds a `DressiAD` graph (pack + GLSL + Vulkan objects), cache hit is
  sendImg → execStep → recvImg on the cached command buffer. Same shape
  keying trick nvdiffrast uses for its GL buffers.
- **External gradient injection**: `BuildBackward` seeds gradient 1 at
  every pixel (build_backward.cpp:87), so the surrogate loss
  `output * seed_leaf` (seed non-requires-grad) makes each leaf gradient
  the exact VJP for `grad_output == seed`. backward() = upload seed →
  execStep → recvImg grads.
- Reactive pruning disabled per engine (`setRebuildCounts(0,0)`): the
  fwd/bwd dirty patterns alternate ({inputs} vs {seed}) and would thrash
  SUBSTAGE repacks. Consequence: a training step costs 2 full graph
  executions (fwd with stale seed + bwd with the real seed); forward-only
  calls use a separate grad-free engine.

## What was done

### C++ core extensions
- `DressiAD(VkContextPtr)` + `DressiAD::createContext()` — one shared
  Vulkan device across all cached engines (was: one instance+device per
  DressiAD).
- `setGradOutputsEnabled` / `inputGrads` — keeps `input_grad_vars` as
  graph roots (the markOutput mechanism) so raw gradients are
  recvImg-readable **without an optimizer**. The previously considered
  identity-optimizer trick is broken: `upd_inp_map[leaf]=leaf` records a
  full-overlap `CopyImage(img, img)` (Vulkan UB). Tests:
  `tests/test_grad_export.cpp` (VJP vs analytic, parity with the
  optimizer-capture path, repeated seed updates, shared context).
- `FaceFetch` exact backward (`n_samples == 0`): per-face screen-bbox scan
  accumulating gy where `idx(p)==f+1` (GLSL variant of the
  `__face_fetch_bwd__` marker + CPU kernel). Needed because the stochastic
  edge-band sampler is badly biased for `interpolate`'s interior-support
  VJP. FD-checked (`Backward.FaceFetchExactBackward`) + GPU parity
  (`GradExport.FaceFetchExactBwdGpuParity`).
- Mesh/topology builders promoted from `examples/common` into the library:
  `include/dressi/mesh_utils.h` (`VertexFacesTex`, `VertexNeighborsTex`,
  `FaceNeighborsTex`, `BuildSoftGeometry`, `UvAsClip`, `ScreenAttr`,
  adjacency builders); examples keep thin using-aliases.
- `CpuImageView` + `sendImg(Variable, CpuImageView)` — borrowed-buffer
  upload without the intermediate `CpuImage` vector copy.

### Transfer-layer performance (the big one)
The eager benchmark started at **35.7 ms/iter @512²**; profiling showed
`recvImg` of the 4 MB rast image took **23 ms**: vkw's memory-type search
picks the first HostVisible|HostCoherent type, which on NVIDIA is
write-combined (uncached) — CPU memcpy *reads* from it run at ~100 MB/s.
Fixes (src/vk/transfer.cpp, context.{h,cpp}):
- Readback staging now requests HostVisible|HostCoherent|**HostCached**
  (fallback to plain coherent when unavailable).
- Persistent grow-on-demand staging buffers per context (send/recv) —
  no per-transfer vkAllocateMemory.
- `RunOneShot` reuses a cached command pool/buffer + fence (was: created
  per transfer).
- Channel-identical images (n_logical == n_phys) skip the re-stride pass
  on both directions; `_stack_batch` avoids the torch.stack copy for N=1.
Result: 35.7 → **4.0 ms/iter @512²** (9x).

### Python layer (`python/`)
- nanobind (`dressi._C`): DressiAD (incl. set_optimizer/add_update),
  Variable (+ graph-building operators), F subset, mesh utils; images
  cross as (H, W, C) float32 ndarrays (numpy out via moved-vector
  capsule, zero-copy in via CpuImageView). GIL released around
  exec/send/recv. `nb::set_leak_warnings(false)` (engine caches
  legitimately live until exit).
- `dressi.torch`: `context.py` (shared device + engine/mesh caches),
  `engine.py` (leaves, surrogate-loss finalize incl. multi-output for
  peels, dirty-skip uploads), `meshdata.py` (faces in BOTH encodings:
  IVEC3 for rasterizers, float VEC3 for gathers; `vtx_faces_tex`; unweld
  index), `provenance.py` (WeakValueDictionary side table: rast → {pos,
  tri, resolution, ctx}, uv → {vtx_uv, rast}; nvdiffrast signatures drop
  data the Dressi backwards need), `convert.py` (GL→Vulkan depth remap
  `z_vk = 0.5(z_gl+w)` and its transpose for pos grads), `ops.py` (the
  autograd.Functions).
- Op graphs:
  - `rasterize`: unwelded corners with constant (u,v) barycentric
    attributes + interpolated (z_gl, w) via `F::Rasterize` (perspective-
    correct) + `F::RasterizeFaceId` → rast=(u,v,z/w,tri_id+1). x/y match
    nvdiffrast at the tensor level (verified); tri_id+1 convention
    matches natively.
  - `interpolate`: per-corner `LookupFaces` → exact `FaceFetch` gather by
    the rast tid channel → barycentric combination. Exact VJP w.r.t. attr.
  - `texture`: inverse-UV table rasterized in-graph from provenance
    (UvAsClip + ScreenAttr); train engines only — forward-only sampling
    needs no provenance.
  - `antialias`: `F::AntiAlias` with per-engine iteration seed; grads for
    color and pos (n_samples=0 exact mode for tests).
  - `rasterize_soft`: `F::SoftClip` → `F::RasterizeSoft` chain with the
    example's in-graph Eq.3 shifted-depth reconstruction between peels;
    returns [N,K,H,W,4] (dist,face_id,depth,cov); sigmoid/Eq.6 composed
    in torch.
- Upload dirty-skip tokens: `(data_ptr, _version, shape)` **plus a strong
  reference** to the source buffer — without the pin, a freed tensor's
  address can be reused by the next allocation with `_version` 0 and the
  token falsely hits (this actually happened: FD tensors alternated into
  the same address and texture gradients came back zero).

### Tests
- C++: 105 (72 cpu / 33 gpu) all green, including the new grad-export and
  FaceFetch-exact suites.
- Python (`tests/python`, 20 tests, also a gpu-labeled ctest entry):
  analytic barycentrics/perspective/depth for rasterize; interpolate
  forward vs manual bary + FD grads (linear → exact); texture nearest
  forward + FD grads on the 1:1 mapping; antialias boundary-only forward,
  color FD, pos-grad cosine vs FD, silhouette optimization convergence
  (offset recovered to ±0.03); rasterize_soft channels, two-layer peeling,
  pos-grad cosine, silhouette convergence; texture-fit E2E; engine-cache
  reuse and upload-skip behavior.

## Measurements (Avocado 406v/682f, fixed view, Adam; ms/iter)

| resolution | eager start | eager final | native fused (Python) | C++ example AA | nvdiffrast CUDA | DRTK |
|---|---|---|---|---|---|---|
| 256²  | -    | 3.3 | 0.55 | 0.49 | 1.11 | 2.12 |
| 512²  | 35.7 | 4.0 | 0.57 | 0.49 | 1.18 | 6.15 |
| 1024² | -    | 9.0 | 0.55 | 0.51 | 1.21 | 19.6 |

- `scripts/dressi_torch_bench.py` (eager) / `scripts/dressi_native_bench.py`
  (fused: whole step incl. in-graph Adam as ONE Dressi graph driven from
  Python — **meets the C++-parity goal**; the remaining ~0.06 ms is the
  Python loop + exec_step call overhead).
- Eager profile @512² after fixes: rasterize 1.0 (read 0.5), AA fwd 1.7,
  AA bwd 1.6 — bounded by per-op CPU round trips and the fwd recompute in
  backward, both inherent to the drop-in eager model with CPU tensors.

## Key decisions & lessons

- nanobind over pybind11: native DLPack/ndarray, no libtorch dependency
  (no per-torch-version rebuilds).
- `setRequiresGradRecursively` propagates DOWNSTREAM (to users) — must be
  called AFTER graph construction; calling setRequiresGrad on a leaf
  before building silently yields an empty backward.
- Identity optimizer for grad readback = self-CopyImage UB; the
  grad-export API is the clean path.
- Data-ptr-based caching of torch tensors requires pinning the source
  buffer (address reuse otherwise forges cache hits).
- Write-combined staging memory is fine to write, catastrophic to read.

## Open issues / next

- **Zero-copy GPU interop (user request)**: direction must be
  Vulkan-allocates → CUDA-imports (VK_KHR_external_memory +
  `cudaImportExternalMemory`; OPAQUE_WIN32 handles on Windows), exposing
  engine images/buffers to torch as CUDA tensors — the reverse (importing
  torch's CUDA allocations into Vulkan) is not possible because torch's
  caching allocator does not export. Needs: exportable-memory allocation
  path in the executor, device matching (LUID), and semaphore or
  wait-idle synchronization; torch-side via `__cuda_array_interface__`
  (cuda-python/cupy) or a small CUDA shim. Non-NVIDIA GPUs keep the CPU
  path as fallback.
- Eager-path headroom: batch per-run uploads into one submit; a
  dirty-pattern-keyed plan cache in the engine would let the backward
  exec skip the clean forward branch (currently recomputed).
- Batches loop per item; range mode unsupported; texture is nearest-only,
  no grad w.r.t. UV (engine limitation, documented).
- Packaging (scikit-build-core wheel) not yet added; the tested dev loop
  is `-DDRESSI_BUILD_PYTHON=ON` + `PYTHONPATH=python`.
- `.venv` (Python 3.13, torch 2.13 CPU) created at the repo root for the
  binding tests; not committed.
