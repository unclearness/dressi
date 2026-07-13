# dressi on Android

The `android/` directory is an Android Studio project bundling all six
examples into one app (`org.dressi.examples`): pick an example, watch
the live view (streams of what the desktop shows in GLFW windows), read the
logs, stop anytime. The engine runs exactly the desktop code path —
headless Vulkan compute + runtime GLSL→SPIR-V (glslang built from source
for arm64).

Verified devices: **Snapdragon 8 Gen 2 (Adreno 740)** and **Dimensity
9500 (Arm G1-Ultra)** — all six examples run to completion on both, with
the same parameters and results as the desktop builds.
`silhouette_optimization` additionally needs the `geometryShader` feature
(checked at startup and grayed out if absent; present on both devices).

## Prerequisites (one-time)

1. **Android Studio** with SDK platform 36, **NDK 28.2.13676358** and
   **CMake 3.31.6** (SDK Manager → SDK Tools; the exact versions are pinned
   in `android/app/build.gradle.kts`).
2. **Datasets**: run a desktop CMake configure once (downloads `data/`):
   ```sh
   cmake --preset msvc          # or your platform preset
   ```
3. **Small env map** (the 97 MB 4k EXR is not bundled into the APK):
   ```sh
   cmake --build --preset release --target downsample_exr
   ./build/tools/Release/downsample_exr.exe data/suburban_garden_4k.exr data/suburban_garden_512.exr 512
   ```

The Gradle task `syncDressiAssets` copies the curated subset
(bunny, Avocado, DamagedHelmet, `suburban_garden_512.exr`) into
`app/src/main/assets/data/` before every build (APK ≈ 15 MB). Missing
files only disable the affected examples at runtime.

## Build & run

- Android Studio: open `android/`, let Gradle sync, Run on a connected
  arm64 device (**no emulator** — the engine needs a real Vulkan GPU).
- CLI:
  ```powershell
  cd android
  $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
  .\gradlew.bat :app:assembleDebug
  adb install -r app\build\outputs\apk\debug\app-debug.apk
  adb logcat -s dressi
  ```

## What to expect on device

- The main screen shows the Vulkan device caps JSON (geometryShader,
  maxImageDimension2D, per-stage descriptor limits) — capture this on a new
  device first.
- The first `execStep` of a run compiles every stage's GLSL on-device
  (seconds on a phone; the log shows stage counts). Steady-state iterations
  follow.
- Example outputs (PNGs, OBJ) land in
  `files/out/<example>/`; pull them with
  ```sh
  adb shell run-as org.dressi.examples ls files/out
  adb shell "run-as org.dressi.examples cat files/out/texture_optimization/recovered_texture.png" > recovered.png
  ```
- The app runs the examples with the desktop-default parameters (only
  the data/output paths differ) — full optimization runs take minutes on
  a phone (e.g. ~5 min texture_optimization, ~15 s silhouette on an
  Adreno 740). Shorten runs by editing `MainActivity.defaultArgs`; a
  quality-gate exit code 1 on a shortened run means "converged less than
  the desktop gate", not an error.

## Cross-device benchmarking

Every example writes a `bench.json` into its output directory on every
run — desktop and Android alike: GPU device name, host info (OS, CPU /
SoC, RAM; on Android also the phone model), the parameters, the median
steady-state ms/iter (warmup excluded), and the example's quality
metric (IoU / accuracy / PSNR / FPS). To build a comparison table
across machines/phones:

```sh
# desktop: run the examples normally; bench.json lands in texopt_out/,
# silopt_out_hardsoftras/ etc.

# per Android device: run the two examples in the app, then pull
mkdir -p bench_results
adb shell run-as org.dressi.examples cat files/out/texture_optimization/bench.json > bench_results/<device>_texopt.json
adb shell run-as org.dressi.examples cat files/out/silhouette_optimization/bench.json > bench_results/<device>_sil.json

python scripts/bench_summary.py texopt_out silopt_out_hardsoftras bench_results
```

Output: one Markdown table per example, one row per device/run, sorted
by median ms/iter. Filenames don't matter — the device name inside the
JSON is what's shown.

## Architecture notes

- `examples/*/run.cpp` hold the example logic
  (`Run<Name>(args, ExampleHost&)`); desktop `main.cpp`s and the Android
  JNI runner both call them. `ExampleHost::makeViewer` returns GLFW windows
  on desktop and "streams" on Android (one SurfaceView, stream selector
  buttons); `cancelled()` is the Stop button.
- The blit path is `ANativeWindow_lock` software copy (no EGL, no second
  Vulkan context) — same design as the desktop `VkViewer` (CPU image →
  window).
- glslang/SPIRV-Tools/SPIRV-Headers/Vulkan-Headers are FetchContent-built
  from the Khronos repos at the lockstep tag `vulkan-sdk-1.4.350.0`
  (matches the desktop SDK; the NDK ships no arm64 glslang and only the C
  Vulkan headers).
