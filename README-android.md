# dressi on Android

The `android/` directory is an Android Studio project bundling all the
examples into one app (`org.dressi.examples`): pick an example, watch
the live view (streams of what the desktop shows in GLFW windows, plus an
"all" button that tiles every stream on the one screen), read the logs,
stop anytime. The engine runs exactly the desktop code path — headless
Vulkan compute + runtime GLSL→SPIR-V (glslang built from source for
arm64).

Verified devices: **Snapdragon 8 Gen 2** (Adreno 740, Xiaomi Pad 6S
tablet), **Dimensity 9500** (Mali-G1-Ultra MC12, Xiaomi 17T Pro) and
**Dimensity 9200+** (Mali-G715 MC11, Xiaomi 13T Pro) — the examples run
to completion with the same parameters and results as the desktop builds
(`shape_texture_optimization`, added later, verified on the Dimensity
9500). Measured per-example timings are in
[docs/benchmarks.md](docs/benchmarks.md).
`silhouette_optimization` and `shape_texture_optimization` additionally
need the `geometryShader` feature (checked at startup and grayed out if
absent; present on all three devices).

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
- CLI (debug — for development/logcat):
  ```powershell
  cd android
  $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
  .\gradlew.bat :app:assembleDebug
  adb install -r app\build\outputs\apk\debug\app-debug.apk
  adb logcat -s dressi
  ```
- **For any performance measurement, build the RELEASE variant instead.**
  The debug build's native `.so` is compiled with no optimization (`-g`
  only, no `-O`/`-DNDEBUG`), so its `execStep` timings are meaningless.
  The `release` buildType is configured as a benchmarking build: native
  is pinned to CMake `Release` (`-O3`/`-DNDEBUG`, matching the desktop
  `release` preset), and it is signed with the debug key and kept
  `debuggable` so `adb install` and `run-as` (bench pull) both work.
  ```powershell
  cd android
  $env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
  .\gradlew.bat :app:assembleRelease
  adb install -r app\build\outputs\apk\release\app-release.apk
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
- The app runs the examples with the desktop-default parameters (only the
  data/output paths differ; `pbr_shading` additionally gets `--frames=200`
  so its interactive orbit self-terminates) — full optimization runs take
  minutes on a phone (e.g. ~5 min texture_optimization, ~15 s silhouette
  on an Adreno 740). Shorten runs by editing `MainActivity.defaultArgs`; a
  quality-gate exit code 1 on a shortened run means "converged less than
  the desktop gate", not an error.

## Cross-device benchmarking

Every example except `shape_texture_optimization` (a two-phase demo with
no single steady-state metric) writes a `bench.json` into its output
directory on every run — desktop and Android alike: GPU device name, host
info (OS, CPU / SoC, RAM; on Android also the phone model), the
parameters, the median steady-state ms/iter (warmup excluded), and the
example's quality metric (IoU / accuracy / PSNR / FPS). To build a
comparison table across machines/phones:

All `adb`/pull commands below run on the **host PC**, not on the phone —
they drive the device over USB. On Windows run them in **Git Bash**
(`adb` is the same binary on every OS); PowerShell works for the plain
`adb` lines too, but `scripts/pull_android_benches.sh` is a bash script
and needs Git Bash / WSL.

```sh
# desktop: run the examples normally; bench.json lands in texopt_out/,
# silopt_out_hardsoftras/ etc.

# per Android device: run the examples in the app (RELEASE build — see
# "Build & run"), then pull every bench.json in one shot. Silhouette runs
# as two technique buttons, each into its own out-dir
# (files/out/silhouette_optimization_hardsoftras and _aa).
scripts/pull_android_benches.sh -o bench_results/<device>
python scripts/bench_summary.py bench_results/<device>
```

The pull script uses `run-as`, so it needs a debuggable build — the
release benchmarking build above qualifies. To pull a single record by
hand instead:

```sh
adb shell run-as org.dressi.examples cat files/out/silhouette_optimization_aa/bench.json > sil_aa.json
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
