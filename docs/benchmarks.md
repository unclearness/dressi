# Benchmarks

Detailed performance data for dressi. See the [README](../README.md) for the
headline summary.

## Methodology

Every figure is the **median steady-state wall time of the example's timed
scope** with a 20-iter warmup excluded — reproduce it exactly; do NOT report a
raw mean-over-all-iters. [Measurement conditions](#measurement-conditions)
defines the scope and every other condition a figure depends on.

- **Exclude a warmup.** The first `execStep` runs the full build (pack + GLSL
  compile + Vulkan objects, ~250 ms) and the reactive cache does a fast rebuild
  at iter 2 and a full rebuild at iter 8. Skip ≥20 iters before timing.
- **Median, not mean.** At ≤512² the wall is dominated by host submit/fence
  latency (GPU work is ~90 µs while the wall is ~0.3 ms), so the mean is pulled
  up by OS-scheduling outliers. Median is the stable metric.
- **CUDA baselines synchronize in blocks.** An eager CUDA pipeline must
  synchronize before/after each timed sample; the Python examples use
  10-iteration blocks after the warmup and report the median block-average.

Every example writes a `bench.json` (GPU device, host os/cpu/ram — phone model
on Android and iOS — parameters, median steady-state ms/iter, warmup, and quality) into
its `--out-dir`; `scripts/bench_summary.py <dirs>` renders the comparison table.

## Measurement conditions

A figure means nothing without its condition. Two figures may be compared only
if they match on viewer condition, timed scope, parameters, and build. The rules
below apply to new measurements; [Current status](#current-status) lists where
the published tables do not meet them yet.

### Viewer condition

Every run is in exactly one of these conditions. "Viewer off" is not a condition:
it has meant both `none` and `idle`.

| condition | viewer object | display update | selected by |
| --- | --- | --- | --- |
| `none` | not created | never | `--no-view=1`, or no display available |
| `idle` | created | never | `--view-interval=0` |
| `live` | created | every iteration | default (`--view-interval=1`) |

- **`none` is the reference condition.** Cross-device tables use it.
- `idle` is not used for new figures. Older rows are `idle`, and whether it
  differs from `none` has not been measured.
- `live` compares only within one platform, because the display path differs:
  FIFO swapchain present on desktop, a CPU blit on Android, `UIImageView` on iOS.
  It moves the median in either direction (`pbr_shading`: M5 Pro 583 to about
  1200 FPS with it off, A14 86 to 74 FPS). Why was not investigated.
- A `--view-interval` other than 0 or 1 is not a benchmark condition.
- A desktop run with no usable display falls back to `none` silently, so the
  condition is what the run actually did, not which flags it was given.

### Timed scope

`median_ms_per_iter` does not measure the same thing in every example. Figures
from different scopes are not interchangeable, and only `execStep` rows are pure
`execStep` time.

| example | scope | timed section |
| --- | --- | --- |
| image_fitting | `execStep` | `execStep` |
| silhouette_optimization | `execStep` | `execStep` |
| pbr_envmap_optimization | `execStep` | `execStep` |
| texture_optimization | `send+execStep` | camera jitter upload + `execStep` |
| pbr_material_optimization | `send+execStep` | camera jitter upload + `execStep` |
| pbr_shading | `send+execStep+recv` | per-frame uploads + `execStep` + frame readback |

### Parameters

Each benchmark runs its defaults, plus only the flags listed here and
`--out-dir`. On mobile the app supplies the data paths.

| label | flags | samples |
| --- | --- | ---: |
| image_fitting | none | 130 |
| texture_optimization | `<data>/bunny` | 1480 |
| silhouette_hardsoftras | `<data>/bunny --technique=hardsoftras` | 130 |
| silhouette_aa | `<data>/bunny --technique=aa` | 130 |
| pbr_material_optimization | none | 1180 |
| pbr_envmap_optimization | none (`--fg-only=1` is the default) | 1480 |
| pbr_shading | `--frames=200` on every platform | 180 |

Add the viewer flag for the condition. Do not add `--snapshot`: it writes PNGs
inside the loop.

### Build and environment

- Release build. Android's debug APK has no optimization.
- `SPDLOG_LEVEL` and `DRESSI_VK_DEBUG` unset: debug logging adds per-stage GPU
  timestamps and `DRESSI_VK_DEBUG` adds validation layers.
- Default packing mode. Do not set `MVK_CONFIG_*` on Apple platforms.
- iOS always writes per-stage timestamps and Apple platforms run with Metal
  argument buffers off. Both are part of those platforms' figures.

### Device state

- One benchmark at a time on the device, and no other GPU workload. Quit apps
  that draw on the GPU, overlays included: on the RTX PRO 6000, a session with a
  game launcher (and its overlay process) and a music player open gave
  image_fitting 0.085–0.095 ms, against 0.058–0.060 ms once they were closed.
- Desktop GPUs run at full performance for the whole run. Some drivers keep a
  GPU with no window in a power-saving state. With no viewer, an RTX 3090 on
  Windows (driver 610.88) ran pbr_envmap at P3 and 780 MHz, 31.1 ms against
  13.8 ms at P0 once a window existed. The RTX PRO 6000 (driver 597.30) stays at
  P0 either way. On NVIDIA under Windows, set "Power management mode" to "Prefer
  maximum performance" for every example exe (NVIDIA Control Panel, Manage 3D
  settings, Program Settings). The Linux counterpart is PowerMizer in
  `nvidia-settings`. With the setting in place, every RTX 3090 run was at P0,
  but in that session the default also reached P0, so it guards against the
  power-saving state rather than being shown to fix it. Check the result:
  `run_benches` samples `nvidia-smi` every 200 ms into `gpu_state.csv` next to
  each `bench.json`, and `bench_summary.py` prints a "GPU state warning" under a
  table when a run spent most of its samples outside P0. Runs with fewer than 3
  samples of the benchmarked GPU are not judged, and other vendors are not
  checked. Intel's Windows driver UI (Intel Graphics Software 1.100) offers no
  such power setting.
- Laptops and tablets: record whether they ran on AC, USB, or battery.
- Mobile: release build, app in the foreground, screen on and unlocked, no
  touches during the run.
- Mobile: a 90 s rest between runs.
- Mobile: after installing the app, run every benchmark once and discard that
  pass. It can move ms/iter as well as warmup: A14 image_fitting gave 1.572 ms in
  the first run after an install and 2.12–2.19 ms in the next five. The cause was
  not investigated, and Android has not been checked.
- Android: thermal status 0 before each run (`adb shell dumpsys thermalservice`).
  `run_ios_benches.py` has no such check on iOS, so only the rest applies there.
- Do not poll the device during a run with `uiautomator` or screen capture.
  Detect completion from logcat or from `bench.json` appearing.

### Repetition and reporting

- Run each benchmark 3 times in one session, deleting `bench.json` first so a
  failed run cannot pass as a result.
- Report the median of the three run medians, and the min–max range next to it.
  One run is not a figure: two runs of one Adreno 740 build gave 37.4 and 41.2 ms
  on texture_optimization.
- Report warmup as the median across runs, taken separately from the ms/iter
  median. The first run after an install can be a cold start: hardsoftras on the
  A14 took 8985 ms in that run and about 1340 ms in the next two.
- Never average run medians, and never mix repetition rules in one table.

### Comparing two builds

- Use the same device, session, condition, and parameters.
- Interleave the builds (A B A B A B) so drift and heat hit both equally.
- Claim a difference only if the two min–max ranges do not overlap.
- Never compare against a stored record or a published table: code, driver, and
  device state have all moved since it was taken.

### Record

`bench.json` must carry what the figure depends on, so that nothing has to be
inferred later:

| key | value |
| --- | --- |
| `viewer` | `none`, `idle`, or `live`, from what the run actually did |
| `timed_scope` | `execStep`, `send+execStep`, or `send+execStep+recv` |
| `build_type` | `Release` or `Debug` |
| `git_commit` | commit of the build, with `-dirty` if modified |
| `driver_version` | Vulkan driver version |
| `fg_only` | pbr_envmap_optimization only |
| `packing_mode` | `rsp` or `naive` |

`bench_summary.py` must not put records with different `viewer` or `timed_scope`
values in one table.

### Current status

Every batch runner measures `none`, with 3 runs per benchmark:
`run_benches` (Windows, Linux, macOS), `run_android_benches.py`, and
`run_ios_benches.py`. The apps stay `live` when started by hand, unless the
Android app's "No viewer" box is ticked.

The [cross-device table](#cross-device--12-vulkan-devices) follows these rules on
all twelve devices: `none`, three runs, a clean Release build, the parameters
above, and a discarded first pass on Android. The builds range from `c218ebb` to
`0152ee3`, but the engine and example code is the same in all of them; the later
commits change only scripts, docs, and the Linux SDK lookup. Where the rows fall
short of the rules:

- The A14 pass came straight after an install, before the discard rule existed.
  Its image_fitting figure is therefore a separate three-run session, 2.162 ms
  (2.130–2.167), where the first pass had given 1.572 ms. The other A14 rows keep
  all three runs, because their first run is not an outlier.
- The RTX PRO 6000 and RTX 6000 Ada were measured before `run_benches` recorded
  the GPU state. A later full rerun on the RTX PRO 6000 with the sampler was at P0
  in every run, and its medians are within 2% of the table (image_fitting 0.057
  against 0.059 ms). The Ada has no GPU-state record.
- `gpu_state.csv` needs `nvidia-smi`, so the RX 9070 XT, UHD 730, and Iris Xe have
  none. On the Linux host, sysfs sampled outside the runner showed the UHD 730 at
  its 1500 MHz maximum while busy, and the RX 9070 XT at 3.34 GHz in 227 of 311
  busy samples and near 1.56 GHz in the rest. Those samples could not be matched
  to examples.
- The Iris Xe did not give stable figures in three sessions (note ¹ under the
  table), so its rows are published with their ranges.
- Some idle apps stayed open: a video editor and two browsers on the RTX PRO 6000,
  Chrome and a remote-desktop session on the Ada and llvmpipe host, VS Code,
  Chrome, and the Copilot app on the RTX 3090, and VS Code on the Iris Xe (the
  measuring session ran inside it).

## Cross-device — 12 Vulkan devices

The identical backward graph runs on four discrete GPUs (three NVIDIA, one AMD),
two Intel iGPUs, Apple M5 Pro, three phones, a tablet, and a CPU software
rasterizer, to the same quality.

Devices:

- **NVIDIA RTX PRO 6000 Blackwell Max-Q** — Ryzen Threadripper 9970X, Win 11 Pro
- **NVIDIA RTX 6000 Ada** — Xeon w7-3465X, Ubuntu 24.04
- **NVIDIA GeForce RTX 3090** — Core i9-10850K, Win 11 Pro (driver 610.88)
- **AMD Radeon RX 9070 XT** — Core i3-14100, Ubuntu 24.04, Mesa 25.2.8 (RADV)
- **Intel Iris Xe Graphics** — Core i7-1370P, Win 11 (integrated GPU, laptop on AC)
- **Intel UHD Graphics 730** — the same Core i3-14100 host as the RX 9070 XT, Mesa
  25.2.8 (ANV; integrated GPU)
- **Apple M5 Pro** — 48 GB, macOS 26.6, MoltenVK 1.4.1
- **Adreno 740** — Snapdragon 8 Gen 2, Xiaomi Pad 6S, Android 16 (tablet)
- **Mali-G1-Ultra MC12** — MediaTek Dimensity 9500, Xiaomi 17T Pro, Android 16 (phone)
- **Mali-G715-Immortalis MC11** — MediaTek Dimensity 9200+, Xiaomi 13T Pro, Android 15 (phone)
- **Apple A14** — iPhone 12 Pro, iOS 18.7, MoltenVK 1.4.1 (phone)
- **llvmpipe (LLVM 20.1.2)** — Mesa lavapipe on the RTX 6000 Ada host, CPU
  software Vulkan

**Vulkan versions.** Built with the Vulkan SDK **1.4.350.0** (1.4.350.1 on the
RX 9070 XT / UHD 730 host; glslang for GLSL compilation; the Android build pins
glslang / SPIRV-Tools / Vulkan-Headers to the matching `vulkan-sdk-1.4.350.0`
tag). The engine creates its instance at API version **Vulkan 1.1** and requests
no feature above it (only the optional `geometryShader` device feature, for
`gl_PrimitiveID`), so any Vulkan 1.1+ driver runs the identical graph. All twelve
devices' drivers report Vulkan 1.1 or newer.

Every row is `viewer: none`, the median of three run medians with their min–max
range, and warmup is the median across runs ([Measurement
conditions](#measurement-conditions)). `packing f→s→st` = unpacked backward-graph
ops → substages → Vulkan render passes. Timed scopes differ by example (`execStep`
for image_fitting, silhouette, and pbr_envmap; `send+execStep` for texture and
pbr_material; `send+execStep+recv` for pbr_shading), so compare within an example
only. The fastest device per example is in bold.

| example | device | median ms/iter | min–max | warmup ms (first) | packing f→s→st | quality |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **image_fitting**<br>64px 150it | RTX PRO 6000 Blackwell | 0.059 | 0.058–0.060 | 212 (210) | 36→8→8 | max err 0.0005 |
|  | RTX 6000 Ada | **0.045** | 0.044–0.048 | 237 (233) | 36→8→8 | max err 0.0005 |
|  | RTX 3090 | 0.086 | 0.083–0.090 | 268 (264) | 36→8→8 | max err 0.0005 |
|  | RX 9070 XT | 0.067 | 0.067–0.067 | 86 (84) | 36→8→8 | max err 0.0005 |
|  | Iris Xe (iGPU)¹ | 0.334 | 0.318–0.426 | 146 (141) | 36→8→8 | max err 0.0005 |
|  | UHD 730 (iGPU)⁴ | 0.132 | 0.099–0.170 | 90 (86) | 36→8→8 | max err 0.0005 |
|  | Apple M5 Pro | 0.285 | 0.277–0.286 | 86 (81) | 36→8→8 | max err 0.0005 |
|  | Adreno 740 (tablet)⁴ | 0.386 | 0.382–0.425 | 332 (246) | 36→8→8 | max err 0.0005 |
|  | Mali-G1-Ultra (phone)⁴ | 0.428 | 0.394–0.484 | 244 (180) | 36→8→8 | max err 0.0005 |
|  | Mali-G715 (phone)⁴ | 2.408 | 2.257–2.700 | 306 (218) | 36→8→8 | max err 0.0005 |
|  | Apple A14 (iPhone)⁵ | 2.162 | 2.130–2.167 | 95 (98) | 36→8→8 | max err 0.0005 |
|  | llvmpipe (CPU) | 0.669 | 0.665–0.682 | 154 (133) | 36→8→8 | max err 0.0005 |
| **texture_optimization**<br>256px 6v 1500it | RTX PRO 6000 Blackwell | 0.956 | 0.954–0.960 | 313 (287) | 232→25→25 | 98.7% |
|  | RTX 6000 Ada | **0.880** | 0.876–0.887 | 362 (347) | 232→25→25 | 98.7% |
|  | RTX 3090² | 2.172 | 1.808–2.223 | 404 (373) | 232→25→25 | 98.7% |
|  | RX 9070 XT | 1.929 | 1.908–1.949 | 163 (154) | 232→25→25 | 98.7% |
|  | Iris Xe (iGPU)¹ | 15.777 | 14.891–19.015 | 343 (312) | 232→27→25 | 98.7% |
|  | UHD 730 (iGPU) | 55.251 | 55.202–55.300 | 218 (251) | 232→25→25 | 98.7% |
|  | Apple M5 Pro | 4.658 | 4.653–4.660 | 180 (139) | 232→38→37 | 98.7% |
|  | Adreno 740 (tablet) | 32.364 | 32.211–32.408 | 1098 (636) | 232→26→25 | 98.7% |
|  | Mali-G1-Ultra (phone) | 23.178 | 22.717–23.446 | 1009 (589) | 232→26→25 | 98.7% |
|  | Mali-G715 (phone) | 19.250 | 19.241–19.375 | 1424 (697) | 232→26→25 | 98.7% |
|  | Apple A14 (iPhone) | 52.582 | 51.524–53.669 | 316 (285) | 232→38→37 | 98.7% |
|  | llvmpipe (CPU) | 127.101 | 126.365–127.572 | 594 (559) | 232→25→25 | 98.7% |
| **silhouette (aa)**<br>128px 8v 150it | RTX PRO 6000 Blackwell | **1.194** | 1.194–1.216 | 655 (509) | 802→38→38 | IoU 0.9529 |
|  | RTX 6000 Ada | 1.238 | 1.235–1.244 | 587 (498) | 802→38→38 | IoU 0.9529 |
|  | RTX 3090² | 2.001 | 1.702–2.097 | 964 (734) | 802→38→38 | IoU 0.9529 |
|  | RX 9070 XT | 2.273 | 2.266–2.276 | 363 (289) | 802→38→38 | IoU 0.9484 |
|  | Iris Xe (iGPU)¹ | 3.758 | 2.854–3.989 | 507 (392) | 802→61→53 | IoU 0.9544 |
|  | UHD 730 (iGPU) | 5.839 | 5.833–5.841 | 420 (335) | 802→38→38 | IoU 0.9543 |
|  | Apple M5 Pro | 2.053 | 2.045–2.096 | 342 (240) | 802→39→38 | IoU 0.9544 |
|  | Adreno 740 (tablet) | 16.332 | 15.386–16.500 | 1762 (942) | 802→53→47 | IoU 0.9447 |
|  | Mali-G1-Ultra (phone) | 13.026 | 12.964–13.092 | 7216 (2071) | 802→53→47 | IoU 0.9550 |
|  | Mali-G715 (phone)³ | 17.748 | 16.068–19.522 | 2777 (1316) | 802→53→47 | IoU 0.9559 |
|  | Apple A14 (iPhone) | 31.659 | 31.353–31.758 | 438 (441) | 802→39→38 | IoU 0.9520 |
|  | llvmpipe (CPU) | 27.747 | 27.743–27.883 | 1296 (636) | 802→38→38 | IoU 0.9473 |
| **silhouette (hardsoftras)**<br>128px 8v 150it | RTX PRO 6000 Blackwell | **2.104** | 2.096–2.138 | 1167 (776) | 3962→123→58 | IoU 0.9514 |
|  | RTX 6000 Ada | 2.930 | 2.929–2.967 | 969 (698) | 3962→123→58 | IoU 0.9514 |
|  | RTX 3090 | 3.588 | 3.500–3.730 | 1751 (1132) | 3962→123→58 | IoU 0.9514 |
|  | RX 9070 XT | 3.917 | 3.888–3.968 | 709 (461) | 3962→123→58 | IoU 0.9494 |
|  | Iris Xe (iGPU)¹ | 14.285 | 11.499–14.481 | 1328 (875) | 3962→202→68 | IoU 0.9491 |
|  | UHD 730 (iGPU) | 9.051 | 8.943–9.082 | 742 (498) | 3962→123→58 | IoU 0.9476 |
|  | Apple M5 Pro | 5.903 | 5.795–5.908 | 741 (464) | 3962→126→61 | IoU 0.9507 |
|  | Adreno 740 (tablet) | 25.307 | 25.033–26.051 | 3615 (1738) | 3962→185→64 | IoU 0.9406 |
|  | Mali-G1-Ultra (phone) | 32.190 | 31.701–32.314 | 4940 (1810) | 3962→174→64 | IoU 0.9542 |
|  | Mali-G715 (phone) | 31.337 | 30.250–32.239 | 6902 (2261) | 3962→174→64 | IoU 0.9512 |
|  | Apple A14 (iPhone) | 29.537 | 29.454–30.008 | 1339 (891) | 3962→126→61 | IoU 0.9502 |
|  | llvmpipe (CPU) | 45.707 | 45.296–45.718 | 1404 (798) | 3962→123→58 | IoU 0.9506 |
| **pbr_material_optimization**<br>albedo 256px tex512 6v 1200it | RTX PRO 6000 Blackwell | **2.749** | 2.698–2.767 | 1365 (990) | 3593→74→42 | PSNR 20.65 dB |
|  | RTX 6000 Ada | 3.523 | 3.493–3.586 | 1160 (899) | 3593→74→42 | PSNR 20.65 dB |
|  | RTX 3090² | 5.539 | 4.803–5.592 | 2177 (1564) | 3593→74→42 | PSNR 20.65 dB |
|  | RX 9070 XT | 4.578 | 4.566–4.731 | 892 (646) | 3593→74→42 | PSNR 20.65 dB |
|  | Iris Xe (iGPU)¹ | 20.427 | 15.739–20.547 | 1465 (1122) | 3593→146→49 | PSNR 20.65 dB |
|  | UHD 730 (iGPU) | 28.360 | 28.336–28.363 | 986 (736) | 3593→74→42 | PSNR 20.65 dB |
|  | Apple M5 Pro | 4.620 | 4.295–4.646 | 1021 (647) | 3593→89→50 | PSNR 20.65 dB |
|  | Adreno 740 (tablet) | 62.445 | 61.027–62.578 | 3601 (1910) | 3593→135→49 | PSNR 20.65 dB |
|  | Mali-G1-Ultra (phone) | 43.633 | 43.585–43.657 | 6346 (2121) | 3593→125→49 | PSNR 20.65 dB |
|  | Mali-G715 (phone) | 45.068 | 44.531–45.333 | 6760 (2564) | 3593→125→49 | PSNR 20.65 dB |
|  | Apple A14 (iPhone) | 56.793 | 53.686–56.872 | 1752 (1343) | 3593→89→50 | PSNR 20.65 dB |
|  | llvmpipe (CPU) | 120.816 | 119.839–121.214 | 2063 (1335) | 3593→74→42 | PSNR 20.48 dB |
| **pbr_envmap_optimization**<br>192px env64 4v 1500it | RTX PRO 6000 Blackwell | 8.641 | 8.608–8.650 | 892 (790) | 2444→43→36 | PSNR 18.11 dB |
|  | RTX 6000 Ada | **8.091** | 8.091–8.107 | 894 (766) | 2444→43→36 | PSNR 18.11 dB |
|  | RTX 3090 | 14.238 | 14.037–14.973 | 1409 (1217) | 2444→43→36 | PSNR 18.11 dB |
|  | RX 9070 XT | 11.424 | 11.404–11.426 | 824 (532) | 2444→43→36 | PSNR 18.12 dB |
|  | Iris Xe (iGPU) | 137.583 | 137.143–138.210 | 1356 (1107) | 2444→55→38 | PSNR 18.11 dB |
|  | UHD 730 (iGPU) | 331.504 | 331.438–331.548 | 748 (908) | 2444→43→36 | PSNR 18.11 dB |
|  | Apple M5 Pro | 45.791 | 45.784–45.802 | 616 (526) | 2444→52→40 | PSNR 18.11 dB |
|  | Adreno 740 (tablet) | 316.315 | 316.126–316.407 | 3015 (2100) | 2444→53→38 | PSNR 18.12 dB |
|  | Mali-G1-Ultra (phone) | 183.925 | 183.705–184.073 | 7541 (3086) | 2444→50→37 | PSNR 18.09 dB |
|  | Mali-G715 (phone) | 202.570 | 202.375–202.616 | 6557 (3013) | 2444→50→37 | PSNR 18.11 dB |
|  | Apple A14 (iPhone)⁶ | 366.912 | 358.063–370.000 | 0 (1224) | 2444→52→40 | PSNR 18.11 dB |
|  | llvmpipe (CPU) | 657.250 | 656.355–658.170 | 2044 (1725) | 2444→43→36 | PSNR 18.12 dB |
| **pbr_shading**<br>512px 200 frames | RTX PRO 6000 Blackwell | 1.190 (840 FPS) | 1.189–1.202 | 381 (363) | 286→5→5 | — |
|  | RTX 6000 Ada | 0.978 (1023 FPS) | 0.977–0.996 | 553 (541) | 286→5→5 | — |
|  | RTX 3090 | 2.425 (412 FPS) | 2.397–2.457 | 565 (548) | 286→5→5 | — |
|  | RX 9070 XT | 1.384 (723 FPS) | 1.343–1.392 | 342 (331) | 286→5→5 | — |
|  | Iris Xe (iGPU)¹ | 6.589 (152 FPS) | 5.876–6.819 | 453 (428) | 286→11→5 | — |
|  | UHD 730 (iGPU) | 5.547 (180 FPS) | 5.530–5.599 | 416 (406) | 286→5→5 | — |
|  | Apple M5 Pro | **0.735** (1360 FPS) | 0.733–0.746 | 206 (187) | 286→5→5 | — |
|  | Adreno 740 (tablet) | 15.843 (63 FPS) | 14.857–16.108 | 811 (659) | 286→11→5 | — |
|  | Mali-G1-Ultra (phone) | 54.462 (18 FPS) | 54.151–55.068 | 512 (687) | 286→10→5 | — |
|  | Mali-G715 (phone) | 23.736 (42 FPS) | 23.509–23.912 | 914 (736) | 286→10→5 | — |
|  | Apple A14 (iPhone) | 13.287 (75 FPS) | 13.040–13.469 | 378 (394) | 286→5→5 | — |
|  | llvmpipe (CPU) | 11.982 (83 FPS) | 11.922–11.983 | 910 (733) | 286→5→5 | — |

¹ **Iris Xe: every row except pbr_envmap is unstable.** Three sessions were run:
on AC while charging, on AC at full charge with rests between passes, and with
every other app quit (the table holds the third, at `0152ee3`). In every session
these rows spread widely between runs: 14–32% min–max in the third, up to 105% in
the second. Pass 1 was the fastest on most rows in the first and third sessions,
but not in the second. pbr_envmap, about 4 minutes per run, stays within 1% (4%
in the second session).
The ACPI thermal zone was logged in the last two sessions and does not follow the
spread. The installed Intel Graphics Software (driver 32.0.101.7085) offers no
power setting, and no GPU frequency log was taken, so the iGPU's clock behaviour
is the untested suspect. Read these rows as ranges.

² **RTX 3090: run 2 was 15–19% faster** than runs 1 and 3 on texture, silhouette
aa, and pbr_material. Every run long enough to sample was at P0
(`gpu_state.csv`), with "Prefer maximum performance" set for the example exes,
and silhouette aa held 1695 MHz in every sample of all three runs, so neither the
P-state nor the SM clock explains it. Not investigated.

³ **Mali-G715 warmed up during its session**: battery temperature rose from 35.5
to 42.2 °C despite the 90 s rests, while thermal status stayed 0. silhouette aa
gave 16.07, 17.75, and 19.52 ms in run order, and image_fitting and hardsoftras
rose with it. Its long rows stay within 2%. The bench started at full charge with
the temperature settled.

⁴ **image_fitting is the shortest row** (150 iterations of under 3 ms), and these
devices spread 11–54% on it: Adreno 740 11%, Mali-G715 18%, Mali-G1-Ultra 21%,
UHD 730 54%. The UHD 730 lands on a few discrete levels: 13 more runs gave 0.099,
0.104, 0.132, 0.170, and 0.214 ms, while the RX 9070 XT on the same host gave
0.066–0.067 every time.

⁵ A separate three-run session. The main A14 pass followed an install, and its
first image_fitting run gave 1.572 ms ([Device state](#device-state)).

⁶ The A14 has no warmup figure: its frames get slower during the 9-minute run
(358 to 370 ms across the three runs, presumably thermal), so the steady-state
median exceeds the warmup iterations and the estimate clamps to zero. The
parenthesized first build is still valid.

The Apple rows include two workarounds: Metal argument buffers are off (about
20% on hardsoftras silhouette), and iOS writes a GPU timestamp after every stage
(about 1 ms on the shortest frames).

Observations:

- **Quality is device-independent.** The metric matches across all twelve
  devices to within stochastic noise: texture 98.7% everywhere, pbr_material
  20.65 dB (llvmpipe 20.48), pbr_envmap 18.09–18.12 dB, silhouette IoU
  0.941–0.956. The Adreno 740 has the lowest IoU on both techniques (0.9447 /
  0.9406). On aa the Mali GPUs pack identically to it (802→53→47) yet reach
  0.955–0.956; the cause was not investigated.
- **Packing follows the driver, not the vendor.** The func count is identical,
  but each driver's Vulkan limits give a different substage count. The NVIDIA
  cards, the RX 9070 XT, the UHD 730 under Mesa, and llvmpipe fuse hardest
  (pbr_material →74), while the Iris Xe under Intel's Windows driver fuses least
  (→146); the phones sit between (Mali →125, Adreno →135). M5 Pro and A14 pack
  identically (→89), with more render passes than the others on texture (→37
  against 25). Metal's 16 sampler slots, which the packer honors, are the likely
  reason. The render-pass count stays low everywhere.
- **No phone wins every row.** The Mali-G1-Ultra (Dimensity 9500) leads the
  mobile devices on silhouette aa, pbr_material, and pbr_envmap; the older
  Mali-G715 (Dimensity 9200+) on texture; the Adreno 740 on hardsoftras and
  image_fitting; the A14 on pbr_shading (75 FPS against 63 on the Adreno and 18
  on the Mali-G1-Ultra). The A14 is also the slowest of them on pbr_envmap,
  texture, and silhouette aa. The Mali GPUs pay the largest first builds, with
  warmup up to 7.5 s against at most 3.6 s on the Adreno and 1.8 s on the A14.
- **The RX 9070 XT sits with the discrete NVIDIA cards**: behind the RTX PRO
  6000 and RTX 6000 Ada on every row, ahead of the RTX 3090 on image_fitting,
  texture, pbr_material, pbr_envmap, and pbr_shading, behind it on both
  silhouette rows.
- **Apple Silicon is close to the discrete cards on some rows only.** The M5 Pro
  is 1.7× behind the RTX PRO 6000 on silhouette aa and pbr_material, on a par
  with the RTX 3090 and RX 9070 XT there, but 4.9× behind on texture and 5.3× on
  pbr_envmap. It is the fastest device on pbr_shading (1360 FPS against 1023 on
  the RTX 6000 Ada). That row includes per-frame uploads and a 512² readback, and
  the NVIDIA cards do not rank there as on the other rows (RTX 6000 Ada 1023, RTX
  PRO 6000 840, RTX 3090 412 FPS), so it is probably limited by host and
  transfer cost, where unified memory helps. No breakdown was taken to confirm
  it, and it says nothing about raw GPU speed.

## vs the paper (Table 4)

One-view **Avocado** (406 v / 682 f) self-reconstruction, median `execStep`.
Hardware: RTX PRO 6000 (Blackwell); the paper used an RTX 2080 (~10× less raw
throughput).

| resolution | paper (RTX 2080) | ours HardSoftRas K=1 | ours K=3 | ours AA |
|---|---|---|---|---|
| 256²  | 0.304 | 0.34 | 0.77 | 0.24 |
| 512²  | 0.442 | 0.38 | 0.88 | 0.24 |
| 1024² | 1.034 | 0.50 | 1.30 | 0.25 |
| 2048² | 3.301 | 1.02 | 2.95 | 0.31 |
| 4096² | —     | 3.08 | 9.64 | 0.51 |

The scaling shape matches the paper: fixed per-pass overhead dominates at low
resolution; pixel work takes over above 1024². At ≤512² the wall is bound by
host submit/fence latency, not GPU work.

### Same sweep on Apple devices

Same Avocado task, `silhouette_optimization --mesh=<Avocado.gltf> --views=1
--screen=<res> --iters=300 --no-view=1`. M5 Pro is the median of three runs, A14
the mean of two.

| resolution | M5 Pro HSR K=1 | M5 Pro K=3 | M5 Pro AA | A14 HSR K=1 | A14 K=3 | A14 AA |
|---|---|---|---|---|---|---|
| 256²  | 0.60 | 1.14 | 0.44 | 5.22 | 10.4 | 5.41 |
| 512²  | 1.05 | 1.99 | 0.48 | 11.0 | 25.4 | 6.88 |
| 1024² | 1.14 | 2.67 | 0.59 | 27.4 | 72.5 | 11.1 |
| 2048² | 2.60 | 7.55 | 0.96 | 86.8 | OOM | 20.2 |
| 4096² | 8.60 | 25.8 | 2.31 | OOM | OOM | 33.9 |

**OOM:** the iPhone cannot build these. K=3 at 4096² fails with
`VK_ERROR_OUT_OF_DEVICE_MEMORY`; K=3 at 2048² and K=1 at 4096² are killed with
signal 9 while building, which is presumed to be the same memory limit (no
jetsam report was written). AA runs at every size.

The M5 Pro's HardSoftRas numbers at 512² and 1024² vary by up to about 45%
between runs.

## vs nvdiffrast and DRTK

Same GPU, same task (`scripts/nvdiffrast_bench.py`, `scripts/drtk_bench.py`).
nvdiffrast 0.4.0 dropped the OpenGL rasterizer, so the paper-era GL backend is
measured on a pinned v0.2.6; [DRTK](https://github.com/facebookresearch/DRTK)
runs its canonical `rasterize` + `render` + `edge_grad_estimator` silhouette
pipeline.

| resolution | nvdiffrast GL v0.2.6 | nvdiffrast CUDA v0.4.0 | DRTK 0.1.0 | ours HSR K=1 | ours AA |
|---|---|---|---|---|---|
| 256²  | 1.78 | 1.11 | 2.12 | **0.15** | 0.49 |
| 512²  | 1.76 | 1.18 | 6.15 | **0.18** | 0.49 |
| 1024² | 1.82 | 1.21 | 19.6 | **0.30** | 0.51 |
| 2048² | 1.70 | 1.28 | 67-108 | 0.83 | **0.57** |
| 4096² | 3.47 | 3.28 | ~220 | 2.87 | **0.78** |

Both techniques beat every baseline at every resolution. nvdiffrast and DRTK need
CUDA, so this comparison has no Apple figures; the sweep below covers dressi on
its own.

- **Why dressi wins below 2048²:** every nvdiffrast backend measures flat
  (~1.1-1.8 ms) here — at this mesh size a PyTorch iteration is bound by
  host-side kernel launches, not GPU work. dressi bakes the whole
  forward+backward+optimizer into pre-recorded Vulkan command buffers with zero
  per-iteration host involvement, so it runs at the cost of the GPU work itself.
- **Why the paper's "gap grows with resolution" doesn't reproduce as-is:** on
  the RTX 2080 nvdiffrast's 2048² step was GPU-bound (5-8 ms); on Blackwell the
  same pixel work costs well under 1 ms and hides beneath the host-launch floor
  until 4096². Same advantage source, saturating earlier.
- **AA is nearly resolution-independent** (0.49 → 0.78 ms for a 256× pixel
  increase): its 8-neighbor forward early-outs on equal triangle IDs
  (perimeter-bound), and its stochastic vertex backward is O(faces).
- **DRTK is pixel-bound from 512² up** — a brute-force CUDA rasterizer with no
  hardware rasterization; silhouette-only gradients deform the mesh into
  transient large/degenerate triangles that multiply covered-pixel cost.
- **Rematerialization closed the former 4096² gap:** the HardSoftRas backward
  used to materialize ~59 fp32 intermediates per stage as attachments; the
  packer's clone pass recomputes cheap forward chains in consumer subpasses
  instead, dropping the crossing count ~40% (4.37 → 2.87 ms at 4096²).

Quality on this benchmark: self-reconstruction reaches silhouette IoU 0.986
(HardSoftRas, K=3) / 0.977 (AA); the stochastic backwards trade a little
per-iteration accuracy for the O(faces) cost and recover it with iterations and
`--samples` / `--peels`, exactly as in the paper.

## PyTorch (`dressi.torch`) performance

Same Avocado silhouette benchmark (median steady-state ms/iter; fixed view,
Adam on positions). The **native fused** column drives the *same* dressi graph
as the C++ `silhouette_optimization` example and lands within noise of it (C++
0.26 ms @512²) — Python is not magically faster; both run the compiled graph
with zero per-iteration PCIe traffic:

| resolution | dressi.torch (eager) | native fused (Python ≈ C++) | nvdiffrast CUDA | DRTK |
|---|---|---|---|---|
| 256²  | 2.3 | **0.21** | 1.11 | 2.12 |
| 512²  | 3.4 | **0.22** | 1.18 | 6.15 |
| 1024² | 8.5 | **0.22** | 1.21 | 19.6 |

Multi-view silhouette example (8 views, 128², 300 iters, CUDA tensors): aa
39 → **8.0 ms/iter**, hardsoftras 97 → **21.6 ms** (nvdiffrast aa on the same
example code: **3.1 ms**). Profiling corrected an earlier assumption: the eager
"host glue" is not PCIe-bound (CUDA↔CPU copies are only ~0.3 ms) — it is per-op
Python/submit overhead (transfer+`execStep` submit/fences, staging memcpy, torch
autograd plumbing), which the native-fused path (one graph, one submit) avoids
entirely. The eager engine batches N-item calls into ONE engine (one execStep
per forward/backward for a whole minibatch), and the per-vertex gather backwards
run WIDE ({V, max_deg} partials + column-sum) — the lever that also sped up the
C++ paths (AA 2.9 → 0.44 ms on an 8-view batch).

On Apple M5 Pro, CPU tensors, same benchmark:

| resolution | dressi.torch (eager) | native fused (Python) | C++ `silhouette_optimization` AA |
|---|---|---|---|
| 256²  | 2.3 | 0.42–0.61 | 0.44 |
| 512²  | 1.9–3.0 | 0.46–0.69 | 0.48 |
| 1024² | 3.6 | 0.54–0.87 | 0.59 |

The native fused bench runs in two speed modes on this machine. Its fast mode
(0.42 / 0.46 / 0.54) is within noise of C++, and C++ also hit the slow mode in
some runs (0.58 at 256², 0.72 at 512²). Why it switches was not investigated.
Multi-view silhouette example (8 views, 128², 300 iters, CPU tensors): aa
3.88 ms/iter, hardsoftras 18.57 ms. These do not compare directly with the CUDA
tensor figures above.

`dressi_native_bench.py` and `dressi_torch_bench.py` now report the median. The
NVIDIA figures above predate that and are means.

## Reproduce

Every runner writes `<out_root>/run<k>/<label>/bench.json`. Build from a clean
commit: a modified tree records `-dirty` in `git_commit`.

```sh
# Windows (NVIDIA: set the power mode in "Device state" first)
python scripts/run_benches.py -o bench_results/<device>

# Linux: Ninja puts the binaries in build/examples, with no Release/ level
python3 scripts/run_benches.py -b build/examples -o bench_results/<device>

# llvmpipe (CPU software Vulkan): select the lavapipe ICD
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json python3 scripts/run_benches.py -b build/examples -o bench_results/llvmpipe

# Android: install the RELEASE build, keep the device awake and unlocked
python scripts/run_android_benches.py -s <serial> --reps 1 -o out/discard   # after an install
python scripts/run_android_benches.py -s <serial> -o bench_results/<device>

# all: one table per example and condition
python scripts/bench_summary.py bench_results
```

### macOS

Plug the Mac into AC power, quit other apps, and run from a clean checkout:

```sh
git fetch origin && git switch feat/moltenvk-macos-ios && git pull
git submodule update --init --recursive
git status --short                    # must print nothing
cmake --preset macos                  # also downloads the datasets
cmake --build --preset macos-release
ctest --preset macos-release
python3 scripts/run_benches.py -b build/examples -o bench_results/<device>
python3 scripts/bench_summary.py bench_results/<device>
```

Binaries have no `Release/` level, hence `-b build/examples`.

### iOS

Do the macOS steps through `macos-release` first, plus the one-time
`suburban_garden_512.exr` step and signing setup in
[README-ios.md](../README-ios.md). Then rebuild and reinstall the app: a build
from before `--no-view` existed takes that flag for a mesh path in
`pbr_shading`.

```sh
cmake --preset ios-device -DDRESSI_IOS_TEAM=<team id>
xcodebuild -project build-ios-device/dressi.xcodeproj \
    -target DressiTests -target DressiExamples -configuration Release \
    -sdk iphoneos -allowProvisioningUpdates
xcrun devicectl device install app --device <udid> \
    build-ios-device/ios/Release-iphoneos/DressiExamples.app
python3 scripts/run_ios_benches.py --device <udid> --reps 1 -o out/discard   # after an install
python3 scripts/run_ios_benches.py --device <udid> -o bench_results/<device>
python3 scripts/bench_summary.py bench_results/<device>
```

Keep the phone unlocked and connected. The app exits between runs, so the home
screen is normal. Add `--avocado` for the
[resolution sweep](#same-sweep-on-apple-devices), which takes much longer.

Keep the discarded pass out of `bench_results`: `bench_summary.py` would count it
as a fourth run.

### Check a result

Before using a table, confirm these in `bench_summary.py`'s output:

- The heading says `viewer: none`.
- `runs` is 3. A failed run is missing from the row, not marked in it.
- `commit` has no `-dirty`.
- No "GPU state warning" follows the table.
- Quality matches the other devices.

`bench_summary.py` needs a real Python (`.venv\Scripts\python` or `uv run
python` on Windows — the bare `python` may hit the Store alias).
