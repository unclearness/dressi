# dressi

A C++/Vulkan unofficial re-implementation of the **Dressi-AD** layer from the
paper *"Dressi: A Hardware-Agnostic Differentiable Renderer with Reactive
Shader Packing and Soft Rasterization"* (Takimoto et al., Eurographics 2022,
[arXiv:2204.01386](https://arxiv.org/abs/2204.01386)).

<!-- GIFs are hosted on the project wiki (demo/ in the wiki repo), not in
     this repository. Regenerate with scripts/make_readme_gifs.sh. -->
<table>
  <tr>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/image_fitting.gif" alt="image fitting" width="100%"><br>
      <sub><a href="examples/image_fitting">image_fitting</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/texture_optimization.gif" alt="texture optimization" width="100%"><br>
      <sub><a href="examples/texture_optimization">texture_optimization</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/silhouette_hardsoftras.gif" alt="silhouette optimization (HardSoftRas)" width="100%"><br>
      <sub><a href="examples/silhouette_optimization">silhouette</a> (HardSoftRas)</sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/silhouette_aa.gif" alt="silhouette optimization (screen-space AA)" width="100%"><br>
      <sub><a href="examples/silhouette_optimization">silhouette</a> (AA)</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/shape_texture_optimization.gif" alt="shape then texture optimization" width="100%"><br>
      <sub><a href="examples/shape_texture_optimization">shape_texture_optimization</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/pbr_shading.gif" alt="PBR shading orbit" width="100%"><br>
      <sub><a href="examples/pbr_shading">pbr_shading</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/pbr_material_optimization.gif" alt="PBR material optimization" width="100%"><br>
      <sub><a href="examples/pbr_material_optimization">pbr_material_optimization</a></sub>
    </td>
    <td align="center" width="25%">
      <img src="https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/pbr_envmap_optimization.gif" alt="PBR environment-map optimization" width="100%"><br>
      <sub><a href="examples/pbr_envmap_optimization">pbr_envmap_optimization</a></sub>
    </td>
  </tr>
</table>

All sequences are live optimization / rendering progress on the Vulkan
engine (every example also runs unmodified on Android and iOS — see below):

📹 **shape_texture_optimization on a phone** — camera capture of the
on-device run (Xiaomi 17T Pro, Dimensity 9500 / Mali-G1-Ultra), showing the
target / prediction / recovered-atlas streams tiled on one screen
([full-quality video](https://raw.githubusercontent.com/wiki/unclearness/dressi/demo/20260718_shape_texture_xiaomi_17tpro_dimensity9500_mali.mp4)):

https://github.com/user-attachments/assets/3107391a-ded4-4245-8d2c-35b3ea4d7f65

Dressi-AD is a define-and-run reverse-mode automatic-differentiation framework
specialized for differentiable rendering: computational graphs of 2D images
are compiled to GLSL fragment shaders, packed into a minimal number of Vulkan
render passes (**Reactive Shader Packing**), and executed on any Vulkan-capable
GPU. Because the backend is portable Vulkan — not CUDA — the *same* graph runs
on a workstation GPU, an integrated GPU, Apple Silicon through MoltenVK, a
pure-CPU software rasterizer, and a phone, to the same numerical quality. The public C++ API follows Appendix A of
the paper.

Vertex-position gradients come from two techniques: **HardSoftRas** from the
Dressi paper, and the differentiable **screen-space anti-aliasing** from
*"Dr.Hair"* (Takimoto et al., CVPR 2024,
[arXiv:2403.17496](https://arxiv.org/abs/2403.17496)). A split-sum **PBR / IBL**
layer (physically based shading, material and environment-map optimization)
builds on the same engine.

### Highlights

- **Hardware-agnostic.** One Vulkan engine, verified across 12 devices — four
  discrete GPUs (three NVIDIA, one AMD), two Intel iGPUs, Apple M5 Pro
  (MoltenVK), three phones (Dimensity 9500 / Mali-G1-Ultra, Xiaomi 17T Pro;
  Dimensity 9200+ / Mali-G715, Xiaomi 13T Pro; A14, iPhone 12 Pro), a tablet
  (Snapdragon 8 Gen 2 / Adreno 740, Xiaomi Pad 6S), and CPU software Vulkan
  (llvmpipe).
- **Faster than the baselines below 2048².** Beats nvdiffrast (GL & CUDA) and
  DRTK at every resolution on the paper's silhouette task, by baking the whole
  forward + backward + optimizer into pre-recorded command buffers with zero
  per-iteration host traffic.
- **Runs on phones, unmodified.** Every example runs on Android and iOS with
  the same parameters and results as desktop (see
  [README-android.md](README-android.md) and [README-ios.md](README-ios.md)).
- **PyTorch drop-in.** `dressi.torch` is an nvdiffrast-compatible API with no
  CUDA requirement.

## Contents

- [Building](#building) — [Windows](#windows), [Linux](#linux),
  [macOS](#macos), [iOS](#ios), [Android](#android), [PyTorch bindings](#pytorch-bindings-dressitorch)
- [Examples](#examples)
- [Performance](#performance) · full tables in [docs/benchmarks.md](docs/benchmarks.md)
- [How it works](#how-it-works) · details in [docs/algorithm.md](docs/algorithm.md)
- [References](#references) · [License](#license) · [Layout](#layout)

## Building

Common requirements: a Vulkan 1.1+ GPU, CMake ≥ 3.28, and the LunarG Vulkan
SDK (used for glslang; auto-detected — see below). The engine core is headless
(`VkContext`); GLFW/glad drive an optional live viewer only.

Build verified with **Vulkan SDK 1.4.350.0** (desktop; the Android build
FetchContent-pins glslang / SPIRV-Tools / Vulkan-Headers to the matching
`vulkan-sdk-1.4.350.0` tag). The engine creates its instance at API version
**Vulkan 1.1**, so it runs on any 1.1+ driver — see
[docs/benchmarks.md](docs/benchmarks.md) for the runtime versions verified
across devices.

### Windows

Visual Studio 2022, x64. The Vulkan SDK is auto-detected under `C:\VulkanSDK\*`
(no `VULKAN_SDK` env var required).

```sh
git clone --recursive <this repo>
cmake --preset msvc                # also downloads the example datasets
cmake --build --preset release
ctest --preset release             # all tests (labels: cpu / gpu)
ctest --preset release -LE gpu     # CPU-only tests (no GPU required)
./build/examples/Release/image_fitting.exe
```

### Linux

Tested on Ubuntu 24.04 (gcc 14 / Ninja). Install the LunarG SDK tarball (no
root needed) under `~/vulkan/<version>/` — it is auto-detected (globs
`~/vulkan/*`, `~/VulkanSDK/*`, `/opt/vulkan/*`; `VULKAN_SDK` still wins). A
system `libvulkan` loader + your GPU's ICD provide the runtime.

```sh
git clone --recursive <this repo>
cmake --preset linux
cmake --build --preset linux-release
ctest --preset linux-release
./build/examples/image_fitting
```

### macOS

Runs on MoltenVK; tested on Apple Silicon (M5 Pro, macOS 26). You need the Xcode
command line tools, CMake and Ninja (for example from Homebrew), and the
[Vulkan SDK](https://vulkan.lunarg.com/) from the LunarG installer. The SDK is
found under `~/VulkanSDK/<ver>/macOS` even if `setup-env.sh` was not sourced,
and the loader is pointed at its MoltenVK at runtime, so no environment
variables are needed.

```sh
xcode-select --install
brew install cmake ninja
git clone --recursive <this repo>
cmake --preset macos               # Ninja, Release; also downloads the datasets
cmake --build --preset macos-release
ctest --preset macos-release       # all tests (labels: cpu / gpu)
./build/examples/image_fitting
./build/examples/texture_optimization data/bunny
```

Binaries sit in `build/examples/`, without the `Release/` level or `.exe` of the
Windows commands below.

### iOS

`ios/` builds two apps with static MoltenVK: `DressiExamples` (every example,
with a live view) and `DressiTests` (all gtest suites). Build on a Mac with
Xcode and the Vulkan SDK, including its iOS component. Run the macOS configure
first: it downloads the datasets that get bundled into the app. The presets use
the Xcode generator because device builds must be signed.

```sh
# one-time: datasets, plus the small env map the app bundles
cmake --preset macos
cmake --build --preset macos-release --target downsample_exr
./build/tools/downsample_exr data/suburban_garden_4k.exr data/suburban_garden_512.exr 512

# device (use the ios-simulator preset and -sdk iphonesimulator for the simulator)
cmake --preset ios-device -DDRESSI_IOS_TEAM=<team id>
xcodebuild -project build-ios-device/dressi.xcodeproj \
    -target DressiTests -target DressiExamples -configuration Release \
    -sdk iphoneos -allowProvisioningUpdates
xcrun devicectl device install app --device <udid> \
    build-ios-device/ios/Release-iphoneos/DressiExamples.app
```

Simulator steps, signing, unattended runs and platform notes:
**[README-ios.md](README-ios.md)**.

### Android

`android/` is an Android Studio project bundling all the examples into one APK
(Kotlin UI + JNI, live view on a SurfaceView, on-device GLSL→SPIR-V). Every
example completes with the same parameters and results as desktop. Build the
**release** variant for meaningful timings (the debug native lib is
unoptimized). Full instructions and on-device notes:
**[README-android.md](README-android.md)**.

```powershell
cd android
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
.\gradlew.bat :app:assembleRelease
adb install -r app\build\outputs\apk\release\app-release.apk
```

### PyTorch bindings (`dressi.torch`)

Python deps are managed with [uv](https://docs.astral.sh/uv/) (`uv sync`
creates `.venv` with CUDA torch; `--extra nvdiffrast` adds the comparison
backend). The bindings themselves need **no** CUDA — they run on any Vulkan GPU.

```sh
uv sync
cmake --preset msvc -DDRESSI_BUILD_PYTHON=ON -DPython_EXECUTABLE=.venv/Scripts/python.exe
cmake --build --preset release --target dressi_py
uv run pytest tests/python
```

On macOS `uv sync` installs the CPU build of torch, and the bindings run on
MoltenVK:

```sh
uv sync
cmake --preset macos -DDRESSI_BUILD_PYTHON=ON -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build --preset macos-release --target dressi_py
PYTHONPATH=python uv run pytest tests/python
```

### Datasets

`data/` is git-ignored (large / licensed) and populated at CMake **configure**
time by [`cmake/FetchAssets.cmake`](cmake/FetchAssets.cmake) (hash-verified,
skipped if present). Pass `-DDRESSI_FETCH_DATA=OFF` to provision `data/`
yourself for offline builds.

| Path | Source | Size |
| --- | --- | --- |
| `data/bunny/` (obj + mtl + atlas) | [kunzhou.net](http://www.kunzhou.net/tex-models/bunny.zip) | ~3 MB |
| `data/Avocado/glTF/` (paper Table 4 mesh) | [glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models/tree/main/2.0/Avocado) | ~8 MB |
| `data/suburban_garden_4k.exr` (PBS env map) | [Poly Haven](https://polyhaven.com/a/suburban_garden) | ~97 MB |

## Examples

The examples ship as native executables (plus Python ports of the
silhouette and texture ones):

```sh
# --- differentiable rendering core ---
./build/examples/Release/image_fitting.exe                       # smoke test
./build/examples/Release/texture_optimization.exe data/bunny     # inverse-UV texture grad
./build/examples/Release/silhouette_optimization.exe data/bunny --technique=hardsoftras
./build/examples/Release/silhouette_optimization.exe data/bunny --technique=aa
./build/examples/Release/silhouette_optimization.exe --mesh=data/Avocado/glTF/Avocado.gltf
./build/examples/Release/shape_texture_optimization.exe data/bunny   # shape, then texture

# --- physically based shading / IBL ---
./build/examples/Release/pbr_shading.exe                         # DamagedHelmet viewer
./build/examples/Release/pbr_material_optimization.exe           # albedo recovery
./build/examples/Release/pbr_envmap_optimization.exe             # environment recovery

# --- PyTorch (nvdiffrast drop-in; same code, either backend) ---
uv run python examples/python/silhouette_optimization.py                       # dressi (Vulkan)
uv run python examples/python/silhouette_optimization.py --backend nvdiffrast  # CUDA
```

A minimal C++ optimization loop:

```cpp
Variable param(VEC3, {64, 64});
Variable target(VEC3, {64, 64});
Variable rendered = param / (param + 1.f);                        // graph of F:: ops
Variable loss = F::Mean((target - rendered) * (target - rendered));
param.setRequiresGradRecursively();

DressiAD ad;
ad.setLossVar(loss);
ad.setOptimizer([](Variables xs, Variables gxs) {                 // SGD
    Variables updated;
    for (size_t i = 0; i < xs.size(); i++)
        updated.push_back(xs[i] - gxs[i] * 0.01f);
    return updated;
});
ad.sendImg(param, init_img);
ad.sendImg(target, target_img);
for (int i = 0; i < 300; i++) ad.execStep();
CpuImage optimized = ad.recvImg(param);
```

## Performance

All figures are the **median steady-state `execStep` wall time** with a 20-iter
warmup excluded (the first steps pay a one-time pack + GLSL-compile + Vulkan
build). At ≤512² the wall is dominated by host submit/fence latency, not GPU
work (GPU timestamps total ~90 µs for HSR at 256² while the wall is ~0.34 ms),
so median (not mean) is the stable metric.

### On-device / cross-device

The identical backward graph runs from a workstation GPU, through AMD and
Intel GPUs and Apple Silicon, down to phones — to the **same quality**.
Median ms/iter of three runs (pbr_shading: FPS), default parameters, no viewer;
the fastest device per example in bold:

| device | image_fitting<br>64px | texture<br>256px | silhouette aa<br>128px | silhouette hardsoftras<br>128px | pbr_material<br>256px | pbr_envmap<br>192px | pbr_shading<br>512px (FPS) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 6000 Blackwell | 0.059 | 0.96 | **1.19** | **2.10** | **2.75** | 8.64 | 840 |
| RTX 6000 Ada | **0.045** | **0.88** | 1.24 | 2.93 | 3.52 | **8.09** | 1023 |
| RTX 3090 | 0.086 | 2.17 | 2.00 | 3.59 | 5.54 | 14.2 | 412 |
| AMD RX 9070 XT | 0.067 | 1.93 | 2.27 | 3.92 | 4.58 | 11.4 | 723 |
| Intel Iris Xe (iGPU) | 0.33 | 15.8 | 3.76 | 14.3 | 20.4 | 138 | 152 |
| Intel UHD 730 (iGPU) | 0.13 | 55.3 | 5.84 | 9.05 | 28.4 | 332 | 180 |
| Apple M5 Pro | 0.29 | 4.66 | 2.05 | 5.90 | 4.62 | 45.8 | **1360** |
| Adreno 740 (tablet) | 0.39 | 32.4 | 16.3 | 25.3 | 62.4 | 316 | 63 |
| Mali-G1-Ultra (phone) | 0.43 | 23.2 | 13.0 | 32.2 | 43.6 | 184 | 18 |
| Mali-G715 (phone) | 2.41 | 19.3 | 17.7 | 31.3 | 45.1 | 203 | 42 |
| Apple A14 (iPhone 12 Pro) | 2.16 | 52.6 | 31.7 | 29.5 | 56.8 | 367 | 75 |
| llvmpipe (CPU) | 0.67 | 127 | 27.7 | 45.7 | 121 | 657 | 83 |
| *quality (all devices)* | max err 0.0005 | 98.7% | IoU 0.94–0.96 | IoU 0.94–0.95 | PSNR 20.65 dB (llvmpipe 20.48) | PSNR 18.1 dB | — |

Quality is device-independent (last row). The phones are Dimensity 9500 /
Mali-G1-Ultra (Xiaomi 17T Pro), Dimensity 9200+ / Mali-G715 (Xiaomi 13T Pro),
and A14 (iPhone 12 Pro); the tablet is Snapdragon 8 Gen 2 / Adreno 740 (Xiaomi
Pad 6S). The Apple rows run on MoltenVK with its Metal argument buffers
disabled, and iOS also writes a GPU timestamp after every stage; both are
workarounds that cost some speed. The Iris Xe figures vary 14–32% between runs
on every example but pbr_envmap. The same figures with ranges, warmup, packing,
and per-device quality are in **[docs/benchmarks.md](docs/benchmarks.md)**.

### vs the paper, nvdiffrast, and DRTK

On the paper's one-view Avocado silhouette task, both techniques **beat every
baseline at every resolution** — at 256² HardSoftRas K=1 is 0.15 ms vs
nvdiffrast CUDA 1.11 ms and DRTK 2.12 ms; at 4096² AA is 0.78 ms vs 3.28 / ~220.
dressi bakes the whole forward + backward + optimizer into pre-recorded command
buffers with zero per-iteration host traffic, so below 2048² — where the PyTorch
baselines are host-launch bound — it runs at the cost of the GPU work itself,
and rematerialization keeps it ahead at 4096² too. The scaling shape matches the
paper (fixed per-pass overhead at low resolution, pixel work above 1024²). Full
tables (vs the paper's Table 4, vs nvdiffrast GL/CUDA and DRTK), the same sweep
on Apple M5 Pro and iPhone 12 Pro, and the discussion:
**[docs/benchmarks.md](docs/benchmarks.md)**. The baselines need CUDA, so they
have no Apple numbers.

## How it works

Each `DressiAD::execStep()` walks the paper's InitStatus ladder — build backward
graph → wire optimizer → traverse → pack substages → pack stages → build Vulkan
objects → submit — and reruns each step only when its inputs changed (a reactive
cache), so a steady loop re-executes pre-recorded command buffers with nothing
crossing PCIe. In brief:

- **Dressi-AD core + Reactive Shader Packing.** Image-op graphs (`Variable` /
  `F::`) compile to GLSL; a greedy packer fuses same-pixel chains into minimal
  Vulkan render passes under each device's limits, then rematerializes cheap
  producers and drops dead outputs. Every op has a CPU kernel checked by finite
  differences. Losses stay images; the optimizer writes updates in-place
  (zero-copy aliasing).
- **Deferred rasterization + texture optimization.** `F::Rasterize` writes
  G-buffer channels; `F::Texture` is differentiable w.r.t. the texture via the
  paper's inverse-UV lookup (with Sobol-jittered sampling for full coverage).
- **Vertex-position gradients.** `F::RasterizeSoft` (HardSoftRas, `--peels=K`)
  and `F::AntiAlias` (Dr.Hair screen-space AA) flow gradients to 3D positions;
  the fully GPU-resident silhouette loop reaches IoU ≈0.95 at the demo
  defaults (≈0.98 with exact gathers and longer runs) with zero
  per-iteration CPU traffic.
- **PyTorch bindings.** `dressi.torch` is an nvdiffrast drop-in backed by the
  Vulkan engine (no CUDA), reconciling eager ⇄ define-and-run with a
  shape-signature engine cache; a native-fused path matches C++ speed from
  Python.
- **PBR / IBL.** Split-sum Cook-Torrance shading with material and
  environment-map optimization; all equirectangular, with exact-transpose
  backwards for the environment ops. The environment is recovered from the
  object's shading and reflections alone (foreground-masked loss — the
  metallic helmet constrains almost the whole sphere).

Full details — every op, the packing / rematerialization passes, the WIDE gather
backwards, the PyTorch engine cache, and known deviations from the paper — are
in **[docs/algorithm.md](docs/algorithm.md)**.

## References

This repository re-implements techniques from the following papers. **If you
use this code in academic work, please cite them** — the Dressi paper for the
renderer as a whole (Dressi-AD, Reactive Shader Packing, HardSoftRas), and
additionally the Dr.Hair paper when you use the screen-space anti-aliasing
vertex gradients (`F::AntiAlias`).

```bibtex
@article{takimoto2022dressi,
  title   = {Dressi: A Hardware-Agnostic Differentiable Renderer with
             Reactive Shader Packing and Soft Rasterization},
  author  = {Takimoto, Yusuke and Sato, Hiroyuki and Takehara, Hikari and
             Uragaki, Keishiro and Tawara, Takehiro and Liang, Xiao and
             Oku, Kentaro and Kishimoto, Wataru and Zheng, Bo},
  journal = {Computer Graphics Forum},
  volume  = {41},
  number  = {2},
  pages   = {13--27},
  year    = {2022},
  doi     = {10.1111/cgf.14455}
}

@inproceedings{takimoto2024drhair,
  title     = {{Dr.Hair}: Reconstructing Scalp-Connected Hair Strands
               without Pre-training via Differentiable Rendering of Line
               Segments},
  author    = {Takimoto, Yusuke and Takehara, Hikari and Sato, Hiroyuki and
               Zhu, Zihao and Zheng, Bo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  pages     = {20601--20611},
  year      = {2024},
  doi       = {10.1109/CVPR52733.2024.01947}
}
```

## License

MIT (see [LICENSE](LICENSE)). Third-party dependencies keep their own licenses:
[VulkanWrapper](https://github.com/unclearness/VulkanWrapper) (MIT, a fork of
[takiyu/VulkanWrapper](https://github.com/takiyu/VulkanWrapper) with Vulkan SDK
1.4.x patches), glad (generated loader), GLFW (zlib), tinyobjloader (MIT), stb
(public domain / MIT), spdlog (MIT), GoogleTest (BSD-3), tinyexr (BSD).

## Layout

```
include/dressi/   public API (Appendix A style)
src/core/         graph nodes, F:: ops, BuildBackward, CPU evaluator, IBL math
src/codegen/      substage -> GLSL fragment/compute shader generation
src/pack/         trivial & greedy (RSP) packers, reactive pruning
src/vk/           headless Vulkan context, executor, CPU<->GPU transfer
src/python/       nanobind module (dressi._C) + dressi.torch eager API
examples/         image_fitting / texture_optimization / silhouette_optimization
                  / shape_texture_optimization / pbr_shading
                  / pbr_material_optimization / pbr_envmap_optimization
android/          one-APK Android app (all examples; Kotlin + JNI)
ios/              iOS examples app + gtest host (UIKit, static MoltenVK)
tests/            GoogleTest suites (ctest labels: cpu / gpu) + tests/python
```
