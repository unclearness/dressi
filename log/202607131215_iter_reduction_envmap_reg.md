# Iteration reduction + envmap darkening (regularizer) + two-device bench

Branch: `cross-device-bench` (PR unclearness/dressi#1). Follows
[202607131115_bench_warmup_recording.md](202607131115_bench_warmup_recording.md)
(warmup recording + `--view-interval`) in the same session.

## Goal

1. The example iteration counts looked padded; reduce each to where its
   quality metric has effectively converged (NVIDIA loss/quality curves).
2. Investigate the envmap optimization darkening ("whole env colors first,
   then black regions grow with iterations") and fix it with a regularizer.
3. Final cross-device perf on BOTH Vulkan devices; update the PR.

## 1. Iteration reduction (data-driven)

Key finding: loss plateaus early but the QUALITY metric keeps climbing,
because it is a different quantity — texture/pbr_material accuracy is
jitter-*coverage*-limited (more iters -> more atlas texels observed) and the
PSNRs climb logarithmically. So reductions are ~2-3x, NOT 10x.

| example | was | now | NVIDIA quality now (was) |
| --- | ---: | ---: | --- |
| image_fitting | 300 | 150 | max_err 0.0005 (0.0000) |
| texture_optimization | 4000 | 1500 | 98.7% (99.4%); 500->96.0 / 1000->98.1 / 2000->99.0 |
| silhouette_optimization | 300 | 150 | IoU 0.951 (0.956); 100->0.939 / 200->0.951 |
| pbr_material_optimization | 2000 | 1200 | PSNR 20.65 dB (21.22); 800->20.2 / 1000->20.4 |
| pbr_envmap_optimization | 1500 | 1500 (kept) | PSNR still climbing at 1500 — see below |

## 2. Envmap darkening — diagnosis + fix

Symptom (NVIDIA, reg off): env PSNR starts 17.78 dB (~uniform-gray init),
DIPS to 14.33 @200, only climbs back to 16.40 @1500 — never beats the init,
while the render loss decreases monotonically (0.076 -> 0.00065). Recovered
env shows salt-and-pepper BLACK, concentrated at the equirect poles (top)
and the sphere's nadir (bottom) — regions the 4 near-horizontal cameras
barely observe.

Root cause (NOT a backward bug — see verification):

- The forward IBL convs are low-pass; their exact transpose is also
  low-pass, so high-freq env components sit in the OBSERVATION NULL SPACE
  (no gradient information). A correct backward cannot recover them.
- Those unconstrained texels develop large null-space compensating
  oscillations. `F::Max(env - step, 0)` rectifies the NEGATIVE lobes to
  exactly 0 => black speckle that accumulates over iters.

Backward verified correct: the finite-difference tests
`Backward.IrradianceConvEnvGrad`, `EnvThroughIrradianceChain`,
`EnvThroughPrefilterConvChain`, `EquirectSampleMapGrad` all pass (CheckGrad
= central FD vs analytic, 2% tol). `EnvGradientParity` only checks GPU-vs-CPU
so it was insufficient; the FD tests are the real proof.

Clamp experiment (`--clamp=0`, `--env-reg=0`, added the `env stats` log +
`env_recovered_signed.png` = negatives-as-red): the black speckle (clamp on:
19.6% near-black, 0% negative) turns into RED at the SAME texels (clamp off:
12.2% negative, min -14.75), confirming black = clamp rectifying
negative-going null-space oscillation. Neither is "correct" — the render is
matched either way (loss ~0.0006) but the env is garbage (PSNR -28 dB
uncapped); the problem is genuinely ill-posed, so a PRIOR is required.

Fix: `--env-reg` (default 0.05) adds a mean per-2x2-block-variance penalty
`Mean(AvgPool2x2(env^2) - AvgPool2x2(env)^2)` to the loss (public ops only;
its gradient rides the same Adam step). Pulls each texel toward its local
mean, filling the holes. Sweep 0.02-0.3 all land ~18.25 dB (insensitive);
0.05 default. **env PSNR 16.40 -> 18.26 dB**, speckle gone, now beats init.
pbr_envmap iters kept at 1500 (PSNR still climbing: 17.22@1000, 18.26@1500).

New pbr_envmap flags: `--env-reg=<w>` (0=off), `--clamp=<0|1>` (diagnostic).

## 3. Two-device benchmark (both Vulkan devices, same machine)

lspci/DRI/Intel-ICD probe: this box has NO Intel GPU (Xeon w7-3465X has no
iGPU). The app's "2 physical devices" = NVIDIA RTX 6000 Ada + llvmpipe
(Mesa lavapipe, CPU software Vulkan; `VK_DRIVER_FILES=.../lvp_icd.json`).
The repo's Intel Iris Xe log was a different machine.

NVIDIA rows use the new default iters; llvmpipe rows use reduced iters
(software is ~130-150x slower; median ms/iter is iter-count-independent, so
still comparable — quality on llvmpipe is under-converged and not a fair
quality row). Full table (with warmup) in the PR comment; produced by
`python scripts/bench_summary.py bench_run bench_run_lvp`.

Headline (NVIDIA / llvmpipe median ms/iter): image_fitting 0.047/0.70,
texture 0.875/130, silhouette hsr 2.94/45, aa 1.29/28, pbr_material
3.50/118, pbr_envmap 8.25/669, pbr_shading 989/83 FPS.

## Tests

CPU 89/89 and GPU 39/39 green after all changes (examples-only + the
`bench.{h,cpp}` helper; no engine/core edits).

## Open issues

- image_fitting max_err rose 0.0000 -> 0.0005 at 150 iters (visually
  identical; bump to ~200 if an exact-zero headline is wanted).
- Per-phase warmup breakdown is still `SPDLOG_LEVEL=debug`-only (a DressiAD
  accessor would let bench.json record it cross-device).
- Envmap poles/nadir remain fundamentally under-observed; the regularizer
  hides it well but more views / elevation coverage would constrain it.
