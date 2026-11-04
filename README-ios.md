# dressi on iOS

`ios/` builds two apps with static MoltenVK:

- **DressiExamples** (`org.dressi.examples`): every example, like the Android
  app. Pick one, watch its streams (plus "all"), read the log, stop anytime.
  Outputs land in the app's `Documents/out/<entry>/`.
- **DressiTests** (`org.dressi.tests`): all gtest suites. An iOS binary only
  runs as an app, so this hosts what `ctest` runs on the desktop.

The engine is the desktop code path: runtime GLSL → SPIR-V (glslang built from
source) → MoltenVK → Metal. Requires iOS 16+ on arm64.

## Status

| target | tests | examples |
| --- | --- | --- |
| iPhone 12 Pro (A14, iOS 18.7) | 132/132 | all 8 entries, default parameters |
| iOS Simulator | 132/132 | 6 of 8 entries; see below |

Median steady-state on the iPhone 12 Pro: image_fitting 2.16 ms/iter,
texture_optimization 57.3, silhouette (hardsoftras) 29.2, silhouette (aa) 32.2,
pbr_shading 13.3 ms/frame (75 FPS), pbr_material_optimization 57.3,
pbr_envmap_optimization 412. Quality matches the desktop builds.

In the simulator `silhouette_optimization_aa` and `shape_texture_optimization`
stop with "image extent exceeds maxImageDimension2D". Simulator Metal caps
textures at 8192, and the bunny's `{V, max_deg}` adjacency table is 16323 wide;
only `{N,1}` tables fold into 2-D tiles. Devices allow 16384 (A11 and later).
Tiling those tables would lift it; `--mesh=<Avocado.gltf>` avoids it for the aa
technique.

## Prerequisites (one-time)

1. Xcode, and the [Vulkan SDK](https://vulkan.lunarg.com/) for macOS with its
   iOS component (the `MoltenVK.xcframework` under `lib/`).
2. Datasets, via a macOS desktop configure, plus the small env map (the 97 MB
   4k EXR is not bundled):
   ```sh
   cmake --preset macos
   cmake --build --preset macos-release --target downsample_exr
   ./build/tools/downsample_exr data/suburban_garden_4k.exr data/suburban_garden_512.exr 512
   ```

## Simulator

```sh
cmake --preset ios-simulator
xcodebuild -project build-ios-simulator/dressi.xcodeproj \
    -target DressiTests -target DressiExamples -configuration Release \
    -sdk iphonesimulator CODE_SIGNING_ALLOWED=NO

xcrun simctl boot "iPhone 17 Pro"
xcrun simctl install booted build-ios-simulator/ios/Release-iphonesimulator/DressiTests.app
xcrun simctl launch --console-pty booted org.dressi.tests
xcrun simctl install booted build-ios-simulator/ios/Release-iphonesimulator/DressiExamples.app
xcrun simctl launch --console-pty booted org.dressi.examples --run-all
```

## Device

```sh
cmake --preset ios-device -DDRESSI_IOS_TEAM=<team id>
xcodebuild -project build-ios-device/dressi.xcodeproj \
    -target DressiTests -target DressiExamples -configuration Release \
    -sdk iphoneos -allowProvisioningUpdates

xcrun devicectl device install app --device <udid> \
    build-ios-device/ios/Release-iphoneos/DressiTests.app
xcrun devicectl device process launch --console --device <udid> org.dressi.tests
```

A free developer account allows three apps on a device at a time. Bundle ids
can be changed with `-DDRESSI_IOS_BUNDLE_PREFIX=<prefix>`.

## Unattended runs

DressiExamples takes launch arguments: `--run=<entry>` runs one list entry and
exits with its code, `--run-all` runs them all. Any other `--option` goes to
the example, e.g. `--run=silhouette_optimization_aa --iters=50`. DressiTests
takes the usual `--gtest_*` flags and writes `Documents/gtest.xml`.

## Platform notes

- Every stage writes a GPU timestamp on iOS. Without them
  `pbr_envmap_optimization` dies with "GPU Hang Error" although its frame takes
  only about 300 ms and no pass over about 100 ms. Why this helps is inferred,
  not established. It costs about 1 ms on the shortest frames.
- The app stops a running example when it goes to the background: iOS forbids
  GPU work there.
- Examples and tests run on 64 MiB-stack threads; the 512 KiB default for
  secondary threads is too small for glslang on large fused shaders.
- `gl_PrimitiveID` works through Metal even though MoltenVK reports
  `geometryShader = false`, so `silhouette_optimization` and
  `shape_texture_optimization` are not gated as they are on Android.
