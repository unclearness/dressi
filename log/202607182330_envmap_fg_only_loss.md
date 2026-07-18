# pbr_envmap_optimization: foreground-only loss by default

## Goal

The user flagged that the envmap demo's loss included the background
pixels, where the optimized env is observed DIRECTLY via
EquirectSample(env, view_ray) — against the spirit of an inverse-rendering
demo ("recover the environment from the object's appearance").

## What was done

- `--fg-only` option added: multiplies the per-view loss diff by the
  G-buffer coverage mask (`pred.mask`, already exposed by PbrOutputs), so
  the env is constrained only through the helmet's diffuse shading and
  specular reflections. **Now the default** (`--fg-only=0` restores the
  background observation for comparison).
- Measured (RTX PRO 6000, defaults 192px / env 64x32 / 4 views / 1500 it):
  fg-only **18.11 dB** vs background-inclusive **18.26 dB** — only
  ~0.15 dB cost, because the mostly-metallic helmet's reflections plus the
  alternating-elevation views constrain almost the whole sphere. Slightly
  more black speckle in weakly-reflected regions (env-reg prior covers the
  rest). ms/iter unchanged (~8.5-9.0): the background branch still renders
  and its transpose stage still runs; only the loss mask changes.
- Docs updated: README table (18.1 dB) + How-it-works wording,
  docs/benchmarks.md footnote ² (device rows keep the recorded
  background-inclusive 18.23-18.26 values), docs/algorithm.md example
  description, CLAUDE.md envmap section, run.cpp header comment
  (gradient-path list reordered, bg path marked `--fg-only=0` only).
- GIF source data + `out/gifs/pbr_envmap_optimization.gif` regenerated
  with the new default (needs re-upload to the wiki `demo/`); Android
  release APK rebuilt and installed.

## Gotcha

First data-gen run after flipping the default silently used the stale
binary (forgot to rebuild) and reproduced the old numbers exactly
(0.000744 / 18.26) — caught because the values matched the baseline to
the last digit. Rebuild → 0.000718 / 18.11.

## Open

- Cross-device rows in docs/benchmarks.md predate the default flip; next
  device re-bench should record the fg-only metric (expect ~18.1).
