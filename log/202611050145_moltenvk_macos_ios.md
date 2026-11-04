# macOS and iOS through MoltenVK

## Goal

Run every test and example on macOS and iOS via MoltenVK. Whatever does not
run gets a reason and a proposed fix.

## Result

Environment: Apple M5 Pro, macOS 26.6.2, Xcode 27.2, Vulkan SDK 1.4.350.0
(MoltenVK 1.4.1), iOS Simulator 27.0.

| target | tests | examples |
| --- | --- | --- |
| macOS | 133/133 (ctest, incl. 20 pytest) | all 7, plus python examples |
| iOS Simulator | 132/132 | 6 of 8 app entries |
| iPhone 12 Pro (A14) | build + sign only | build + sign only |

The device was not run: the free developer profile allows three apps and the
phone holds three others (`cross3dgs`, `.e2e`, `.sfmbench`). They were left
alone. Removing one, or a paid account, unblocks it.

Simulator failures: `silhouette_optimization_aa` and
`shape_texture_optimization` hit its 8192 texture cap (`{V, max_deg}` adjacency
width 16323, not tile-aware). Devices have 16384. aa itself passes there with
Avocado (IoU 0.9827).

## What broke, in order found

- libc++ rewrites `std::less<Key>` in map/set to `std::less<void>`, skipping the
  `std::less<Variable>` specialization and calling the graph-building
  `operator<`. Replaced with `VariableLess` / `VarMap` / `VarSet`.
- ld64 has no `--start-group`. The loader needs the ICD manifest pointed at
  (`PrepareVulkanLoaderEnv`, compiled-in SDK path, environment wins).
- `silhouette --technique=hardsoftras` went NaN. Aligned 8x8 NaN blocks, RSP
  only, not Naive. Three misleading fixes first:
  - fast-math off "fixed" 128^2 by luck, 256^2 still failed;
  - capping samplers at 16 was correct but insufficient (fused shaders used up
    to 25; Metal has 16 slots, and the packer only looked at sampled images);
  - capping attachments per render pass looked like it worked, with no sensible
    threshold (16 fine, 24 broken).
  The simulator gave the real cause: RSP fusion mixes input attachments (bound
  directly) with descriptors in a Metal argument buffer, which MoltenVK
  misreads. Naive never mixes them. `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS=0`
  fixes macOS and the simulator (20 failures -> 0).
- Metal's per-pixel color attachment budget: the simulator asserted "requires
  40 bytes ... supports 32". Vulkan has no such limit; added
  `max_color_bytes_per_pixel` to the packer.
- Simulator's real texture limit (8192) is below what MoltenVK reports.

## Decisions

- Argument buffers off instead of limiting stages to one subpass. It explains
  every symptom, while the subpass limit only hid them. With it, one-subpass and
  chained packings agree bit-for-bit (loss 0.11321039); with argument buffers on
  they did not match the off value (0.10796), so results were likely quietly
  wrong even without NaNs. Costs about 20% on the hardsoftras silhouette
  (4.8 -> 5.8 ms/iter), accepted for correctness.
- Limits are queried at runtime from the MTLDevice GPU family
  (`src/vk/metal_limits.mm`), not per-platform constants: Apple4+ 64 B, Apple2
  32 B and 8192 (the simulator reports Apple2), Mac2 unlimited. The executor
  asserts the color-byte and sampler limits; the packers are expected to honor
  them, but past them Metal corrupts or aborts instead of failing cleanly.
- Fork branch `dressi-moltenvk` on `dressi2-sdk-compat`: cherry-picks the
  existing portability commit from `apple-portability` (0b4a26d) without that
  branch's unrelated subgroup / SPIR-V 1.3 commits, then adds iOS support.

## Measurements

Median steady-state ms/iter, Apple M5 Pro, viewer off, final configuration.

| example | ms/iter | quality |
| --- | --- | --- |
| image_fitting | 0.321 | max err 0.0005 |
| texture_optimization | 4.686 | 98.7% |
| silhouette (aa) | 2.070 | IoU 0.9544 |
| silhouette (hardsoftras) | 6.205 | IoU 0.9507 |
| pbr_shading | 0.756 (1323 FPS) | finite |
| pbr_material_optimization | 4.627 | PSNR 20.65 dB |
| pbr_envmap_optimization | 45.73 | PSNR 18.11 dB |

Simulator: pbr_material 14.0, pbr_envmap 49.7, pbr_shading 1.64 ms.
Python on macOS: `dressi_native_bench.py` 0.471 ms @512^2, eager
`dressi_torch_bench.py` 1.780 ms @512^2. Not cross-checked against the C++
Avocado path at matching settings.

## Open issues

- iPhone runs (tests, all examples, A14 budget of 64 B) unverified.
- The MoltenVK mix bug is worked around, not reported or understood at the
  source level. Enabling argument buffers again needs that fix.
- GLFW viewer ran without error on macOS, but screen capture permission was
  missing, so the window contents were not checked by eye.
- The iOS app UI was only exercised through `--run` / `--run-all`.
- `{V, max_deg}` tiling would let the two simulator failures run.
- Windows, Linux and Android were not rebuilt here; `VarMap` and the packer
  changes touch all of them.
