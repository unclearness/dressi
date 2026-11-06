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

- One benchmark at a time on the device, and no other GPU workload.
- Laptops and tablets: record whether they ran on AC, USB, or battery.
- Mobile: release build, app in the foreground, screen on and unlocked, no
  touches during the run.
- Mobile: a 90 s rest between runs.
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

The [cross-device table](#cross-device--10-vulkan-devices) predates these rules
and mixes conditions:

- Android rows are `live`, except where a record predates the viewer. The Adreno
  740 image_fitting figure (0.225 ms) is most likely `none`: that APK was
  installed before the example gained a viewer (bab660f), and the same device
  gives about 0.71 ms with one.
- `pbr_shading` rows are `live`; the other Apple rows are `idle`, and iOS
  silhouette is `idle` where desktop silhouette is `none`.
- `pbr_shading` used 60 frames on desktop and 200 on mobile.
- Repetition varies: one run on most devices, the median of three on the M5 Pro,
  the mean of two on the A14.

## Cross-device — 10 Vulkan devices

The identical backward graph runs across three discrete NVIDIA GPUs, an Intel
iGPU, Apple M5 Pro, three phones, a tablet, and a CPU software rasterizer — to
the same quality.

Devices:

- **NVIDIA RTX PRO 6000 Blackwell Max-Q** — Ryzen Threadripper 9970X, Win 11 Pro
- **NVIDIA RTX 6000 Ada** — Xeon w7-3465X, Ubuntu 24.04
- **NVIDIA GeForce RTX 3090** — Core i9-10850K, Win 11 Pro
- **Intel Iris Xe Graphics** — Core i7-1370P, Win 11 (integrated GPU)
- **Apple M5 Pro** — 48 GB, macOS 26.6, MoltenVK 1.4.1
- **Adreno 740** — Snapdragon 8 Gen 2, Xiaomi Pad 6S, Android 16 (tablet)
- **Mali-G1-Ultra MC12** — MediaTek Dimensity 9500, Xiaomi 17T Pro, Android 16 (phone)
- **Mali-G715-Immortalis MC11** — MediaTek Dimensity 9200+, Xiaomi 13T Pro, Android 15 (phone)
- **Apple A14** — iPhone 12 Pro, iOS 18.7, MoltenVK 1.4.1 (phone)
- **llvmpipe (LLVM 20.1.2)** — Mesa lavapipe, CPU software Vulkan

**Vulkan versions.** Built with the Vulkan SDK **1.4.350.0** (glslang for GLSL
compilation; the Android build pins glslang / SPIRV-Tools / Vulkan-Headers to
the matching `vulkan-sdk-1.4.350.0` tag). The engine creates its instance at
API version **Vulkan 1.1** and requests no feature above it (only the optional
`geometryShader` device feature, for `gl_PrimitiveID`), so any Vulkan 1.1+
driver runs the identical graph. All ten devices' drivers report Vulkan 1.1
or newer — e.g. the primary RTX PRO 6000 Blackwell reports Vulkan **1.4.329**
(NVIDIA 596.72).

Median ms/iter (default parameters), `packing f→s→st` = unpacked backward-graph
ops → substages → Vulkan render passes:

| example | device | median ms/iter | warmup ms (first) | packing f→s→st | quality |
| --- | --- | ---: | ---: | ---: | --- |
| **image_fitting**<br>64px 150it | RTX PRO 6000 Blackwell | 0.056 | 269 (266) | 36→8→8 | max err 0.0005 |
| | RTX 6000 Ada | **0.048** | 288 (284) | 36→8→8 | max err 0.0005 |
| | RTX 3090 | 0.116 | 592 (578) | 36→8→8 | max err 0.0005 |
| | Iris Xe (iGPU) | 0.320 | 136 (127) | 36→8→8 | max err 0.0005 |
| | Apple M5 Pro | 0.324 | 65 (59) | 36→8→8 | max err 0.0005 |
| | Adreno 740 (tablet) | 0.225 | 193 (136) | 36→8→8 | max err 0.0005 |
| | Mali-G1-Ultra (phone) | 1.007 | 171 (123) | 36→8→8 | max err 0.0005 |
| | Mali-G715 (phone) | 2.038 | 345 (256) | 36→8→8 | max err 0.0005 |
| | Apple A14 (iPhone) | 2.38 | 105 (102) | 36→8→8 | max err 0.0005 |
| | llvmpipe (CPU) | 0.729 | 160 (138) | 36→8→8 | max err 0.0005 |
| **texture_optimization**<br>256px 6v 1500it | RTX PRO 6000 Blackwell | 0.958 | 243 (214) | 232→25→25 | 98.7% |
| | RTX 6000 Ada | **0.889** | 362 (347) | 232→25→25 | 98.7% |
| | RTX 3090 | 2.248 | 635 (580) | 232→25→25 | 98.7% |
| | Iris Xe (iGPU) | 15.695 | 338 (293) | 232→27→25 | 98.7% |
| | Apple M5 Pro | 4.663 | 160 (116) | 232→38→37 | 98.7% |
| | Adreno 740 (tablet) | 48.361 | 706 (516) | 232→26→25 | 98.7% |
| | Mali-G1-Ultra (phone) | 23.565 | 1103 (531) | 232→26→25 | 98.7% |
| | Mali-G715 (phone) | 24.255 | 1689 (772) | 232→26→25 | 98.7% |
| | Apple A14 (iPhone) | 52.06 | 383 (274) | 232→38→37 | 98.7% |
| | llvmpipe (CPU) | 129.7 | 614 (561) | 232→25→25 | 98.7% |
| **silhouette (aa)**<br>128px 8v 150it | RTX PRO 6000 Blackwell | **1.213** | 644 (496) | 802→38→38 | IoU 0.9529 |
| | RTX 6000 Ada | 1.261 | 595 (506) | 802→38→38 | IoU 0.9529 |
| | RTX 3090 | 2.813 | 2256 (1749) | 802→38→38 | IoU 0.9529 |
| | Iris Xe (iGPU) | 3.339 | 393 (284) | 802→61→53 | IoU 0.9544 |
| | Apple M5 Pro | 2.049 | 357 (236) | 802→39→38 | IoU 0.9544 |
| | Adreno 740 (tablet) | 21.572 | 1612 (798) | 802→53→47 | IoU 0.9447 |
| | Mali-G1-Ultra (phone) | 11.519 | 8131 (2012) | 802→53→47 | IoU 0.9550 |
| | Mali-G715 (phone) | 21.244 | 2728 (1228) | 802→53→47 | IoU 0.9559 |
| | Apple A14 (iPhone) | 32.55 | 402 (438) | 802→39→38 | IoU 0.9520 |
| | llvmpipe (CPU) | 28.20 | 1352 (666) | 802→38→38 | IoU 0.9473 |
| **silhouette (hardsoftras)**<br>128px 8v 150it | RTX PRO 6000 Blackwell | **2.163** | 1179 (776) | 3962→123→58 | IoU 0.9514 |
| | RTX 6000 Ada | 3.009 | 951 (695) | 3962→123→58 | IoU 0.9514 |
| | RTX 3090 | 6.824 | 2900 (2167) | 3962→123→58 | IoU 0.9514 |
| | Iris Xe (iGPU) | 9.956 | 907 (584) | 3962→202→68 | IoU 0.9491 |
| | Apple M5 Pro | 6.418 | 772 (478) | 3962→126→61 | IoU 0.9507 |
| | Adreno 740 (tablet) | 23.661 | 3502 (1587) | 3962→185→64 | IoU 0.9406 |
| | Mali-G1-Ultra (phone) | 32.455 | 5520 (1794) | 3962→174→64 | IoU 0.9542 |
| | Mali-G715 (phone) | 36.626 | 6576 (2818) | 3962→174→64 | IoU 0.9512 |
| | Apple A14 (iPhone) | 28.88 | 1384 (888) | 3962→126→61 | IoU 0.9502 |
| | llvmpipe (CPU) | 46.91 | 1384 (806) | 3962→123→58 | IoU 0.9506 |
| **pbr_material_optimization**<br>albedo 256px tex512 6v 1200it | RTX PRO 6000 Blackwell | **2.727** | 1362 (964) | 3593→74→42 | PSNR 20.65 dB |
| | RTX 6000 Ada | 3.510 | 1220 (927) | 3593→74→42 | PSNR 20.65 dB |
| | RTX 3090 | 6.042 | 3539 (2661) | 3593→74→42 | PSNR 20.65 dB |
| | Iris Xe (iGPU) | 17.923 | 1019 (788) | 3593→146→49 | PSNR 20.65 dB |
| | Apple M5 Pro | 4.107 | 1018 (613) | 3593→89→50 | PSNR 20.65 dB |
| | Adreno 740 (tablet) | 55.171 | 3786 (1868) | 3593→135→49 | PSNR 20.65 dB |
| | Mali-G1-Ultra (phone) | 40.891 | 7667 (3336) | 3593→125→49 | PSNR 20.65 dB |
| | Mali-G715 (phone) | 46.994 | 7875 (3632) | 3593→125→49 | PSNR 20.65 dB |
| | Apple A14 (iPhone) | 57.41 | 1637 (1338) | 3593→89→50 | PSNR 20.65 dB |
| | llvmpipe (CPU) | 117.7 | 2098 (1353) | 3593→74→42 | PSNR 20.48 dB |
| **pbr_envmap_optimization**²<br>192px env64 4v 1500it | RTX PRO 6000 Blackwell | 8.532 | 1980 (1642) | 2436→43→36 | PSNR 18.26 dB |
| | RTX 6000 Ada | **8.193** | 904 (777) | 2436→43→36 | PSNR 18.26 dB |
| | RTX 3090 | 14.012 | 6494 (4582) | 2436→43→36 | PSNR 18.26 dB |
| | Iris Xe (iGPU) | 105.30 | 833 (792) | 2436→54→38 | PSNR 18.26 dB |
| | Apple M5 Pro³ | 45.78 | 620 (512) | 2444→52→40 | PSNR 18.11 dB |
| | Adreno 740 (tablet) | 304.1 | 3124 (2041) | 2436→53→38 | PSNR 18.26 dB |
| | Mali-G1-Ultra (phone) | 181.1 | 8252 (3800) | 2436→50→37 | PSNR 18.23 dB |
| | Mali-G715 (phone) | 200.2 | 10312 (5076) | 2436→50→37 | PSNR 18.26 dB |
| | Apple A14 (iPhone)³ | 362.0 | — (1232) | 2444→52→40 | PSNR 18.11 dB |
| | llvmpipe (CPU) | 669.0 | 2043 (1757) | 2436→43→36 | PSNR 18.25 dB |
| **pbr_shading**<br>512px 60 frames¹ | RTX PRO 6000 Blackwell | 1.477 (677 FPS) | 305 (289) | 286→5→5 | 677 FPS |
| | RTX 6000 Ada | **1.040** (961 FPS) | 538 (526) | 286→5→5 | 961 FPS |
| | RTX 3090 | 2.948 (339 FPS) | 789 (764) | 286→5→5 | 339 FPS |
| | Iris Xe (iGPU) | 8.660 (116 FPS) | 702 (654) | 286→11→5 | 116 FPS |
| | Apple M5 Pro | 1.714 (583 FPS) | 179 (173) | 286→5→5 | 583 FPS |
| | Adreno 740 (tablet) | 17.717 (56 FPS) | 825 (638) | 286→11→5 | 56 FPS |
| | Mali-G1-Ultra (phone) | 53.746 (19 FPS) | 779 (671) | 286→10→5 | 19 FPS |
| | Mali-G715 (phone) | 24.044 (42 FPS) | 1206 (973) | 286→10→5 | 42 FPS |
| | Apple A14 (iPhone) | 11.67 (86 FPS) | 360 (377) | 286→5→5 | 86 FPS |
| | llvmpipe (CPU) | 11.97 (84 FPS) | 894 (719) | 286→5→5 | 84 FPS |

¹ `pbr_shading` is an interactive viewer; on Android and iOS it runs a
deterministic 200-frame orbit (vs desktop headless 60) so it self-terminates and emits
`bench.json`. The metric is exec+readback per frame, so the frame budget does
not change the median. **All rows have the live viewer on:** the batch script
that recorded the older desktop rows did not turn it off, and the Android app
leaves it on. It sits outside the timed section but still moves the result, in
either direction. With it off, the M5 Pro goes from 583 to about 1200 FPS
(0.74–1.05 ms over 8 runs) and the A14 from 86 to 74 FPS. `run_benches` now turns
it off, so new runs of that script do not compare with this table until the
other rows are measured again.

² measured with the background-inclusive loss (now `--fg-only=0`). The
current default masks the loss to the helmet's coverage — the env is
observed only through the object's shading/reflections — and reaches
18.11 dB on the RTX PRO 6000 at an unchanged ms/iter (the background
branch still renders; only the loss mask differs).

³ measured with the current default (`--fg-only=1`), so the PSNR compares with
the 18.11 dB above, not with the column. The A14 has no warmup figure: its frames
get slower during the 9-minute run (running median 299 ms at iter 200, 360 ms at
the end, presumably thermal), so the steady-state median exceeds the warmup
iterations and the estimate clamps to zero. The parenthesized first build is
still valid.

Apple figures are the median of three runs on the M5 Pro and the mean of two
runs on the A14, with 90 s rests between examples on the phone (`pbr_shading`:
five and three runs). The A14 rows other than `pbr_shading` have the viewer off,
while the Android rows ran with the app's default live view. With the viewer on,
the A14 moves by +20% on texture (62.2 ms), −4% on pbr_material (55.2 ms), and 0
to +7% on image_fitting and silhouette. They include two
workarounds: Metal argument buffers are off (about 20% on hardsoftras
silhouette), and iOS writes a GPU timestamp after every stage (about 1 ms on the
shortest frames). Short frames on the M5 Pro vary between runs: image_fitting
0.26–0.33 ms. The cause was not investigated. The others stayed within 4%.

Observations:

- **Quality is device-independent** — the metric matches across all ten
  devices to within stochastic noise (texture 98.7% everywhere, pbr_envmap
  18.23–18.26 dB with the background-inclusive loss, silhouette IoU 0.941–0.956),
  validating the engine.
- **Packing is device-dependent** — the func count is identical, but each
  device fuses to a different substage count under its own Vulkan limits: the
  NVIDIA cards fuse hardest (pbr_material →74), the Intel iGPU least (→146), the
  phones in between (both Mali →125 / Adreno →135). The render-pass (stage) count
  that costs draw calls stays low everywhere. M5 Pro and A14 pack identically
  (pbr_material →89), with more stages than the other devices on texture and
  pbr_material (texture →37 vs 25, pbr_material →50 vs 42–49). Metal's 16 sampler
  slots, which the packer honors, are the likely reason.
- **Mobile spans the same graph** — the three mobile devices (Xiaomi 17T Pro and
  13T Pro phones, Xiaomi Pad 6S tablet) are ~1-2 orders slower on the heavy
  optimizers but converge to the same quality. The Dimensity 9500 (Mali-G1-Ultra,
  17T Pro) beats the Snapdragon 8 Gen 2 (Adreno 740, Pad 6S) on the heavy
  optimizers yet is slower on light/forward paths and pays a much larger
  first-build warmup — its shader compiler, not steady-state throughput, is the
  bottleneck. The older Dimensity 9200+ (Mali-G715, 13T Pro) trails both on the
  heavy optimizers but leads the Dimensity 9500 on light/forward paths
  (pbr_shading 42 vs 19 FPS) — the same compile-bound signature, one generation
  back.
- **Apple Silicon sits between the iGPU and the discrete cards** on the heavy
  optimizers: M5 Pro is 1.5–5.4× behind the RTX PRO 6000 and 3.4–4.4× ahead of
  Iris Xe on texture and pbr_material. pbr_shading is the exception, ahead of the
  RTX 3090 (583 vs 339 FPS). That row includes per-frame uploads and a 512²
  readback, and the NVIDIA cards do not rank there as on the other rows (RTX 6000
  Ada 961, RTX PRO 6000 677, RTX 3090 339 FPS), so it is probably limited by host
  and transfer cost, where unified memory helps. No NVIDIA breakdown was taken to
  confirm it, and it says nothing about raw GPU speed. The A14, a 2020 phone, is
  the fastest mobile device on pbr_shading (86 FPS) and the slowest on pbr_envmap.

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
# Windows, Linux
python scripts/run_benches.py -o bench_results/<device>

# llvmpipe (CPU software Vulkan): select the lavapipe ICD
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json python scripts/run_benches.py -o bench_results/llvmpipe

# Android: install the RELEASE build, keep the device awake and unlocked
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
python3 scripts/run_ios_benches.py --device <udid> -o bench_results/<device>
python3 scripts/bench_summary.py bench_results/<device>
```

Keep the phone unlocked and connected. The app exits between runs, so the home
screen is normal. Add `--avocado` for the
[resolution sweep](#same-sweep-on-apple-devices), which takes much longer.

### Check a result

Before using a table, confirm these in `bench_summary.py`'s output:

- The heading says `viewer: none`.
- `runs` is 3. A failed run is missing from the row, not marked in it.
- `commit` has no `-dirty`.
- Quality matches the other devices.

`bench_summary.py` needs a real Python (`.venv\Scripts\python` or `uv run
python` on Windows — the bare `python` may hit the Store alias).
