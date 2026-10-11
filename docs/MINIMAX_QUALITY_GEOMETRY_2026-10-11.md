# MiniMax meaningful-resolution quality replay — 2026-10-11

The 32×32 smoke output is blurred and does not establish semantic media quality.
The larger native run is dispatched separately on Spark `.178`:

- Root/unit: `/tmp/lunaflux-minimax-joint-quality128-20261011-v1` /
  `lunaflux-minimax-joint-quality128-20261011-v1.service`.
- Committed media executable: `de9e1ec8`; AOT geometry is 128×128 and five
  seconds, with 1,024 packed denoiser rows.
- Literal caption: `A red cat`, three original tokenizer IDs; seed 17.
- Schedule: twenty evaluations from twenty-one original inclusive-zero F32
  points, all fifty DiT blocks and both original-weight decoder paths.
- Declared device budget: 96 GiB; unit swap limit zero; six-hour timeout.
- Canonical terminal latents are captured before decoder mean/std conversion.

At the most recent check the exact user unit is running, stderr is empty and
unit swap use is zero. The redirected diagnostic stdout is buffered; an empty
log alone is not evidence that the GPU request has not begun. Actual media has
not yet been collected or inspected. No visual or audio quality pass is claimed.

## Geometry-correct original decoder reference

The prior offline VideoVAE bridge hard-coded `[1,24,37,2,2]`, corresponding to
32×32 output. It cannot consume the new `[1,24,37,8,8]` generated latent.
The bridge and MoonBit runner now accept explicit latent time/height/width,
preserve canonical NCTHW F32 bytes, and pass that shape to the installed original
decoder and processor. Default 32×32 behavior remains available. There is no
new production dependency, request-path scan, hashing or deployment work.

An original standardized-zero replay of the **larger shape** completes on Spark
`.179`, exits zero without OOM, releases its GPU and produces
`[1,3,124,128,128]`: **24,379,392 F32 bytes**. This validates reference geometry
and execution, not generated-latent numerical parity or caption fidelity.
Cgroup peak metrics were not retained for this short original decoder run;
the 32-GiB/no-swap Docker limit is not a measurement of complete CUDA residency.

Reference root: `/tmp/lunaflux-minimax-upstream-video-quality128-zero-20261011-v1`.
Source tests pass; the original bridge compiles with `-Wall -Wextra -Werror` in
the installed ARM reference image. After native completion, replay its literal
captured latent rather than substituting the standardized zero used here.

```sh
moon run scripts/run-minimax-upstream-video.mbtx prepare NEW_REFERENCE_ROOT CAPTURED_VIDEO_LATENT_F32 --latent-shape 37 8 8
moon run scripts/run-minimax-upstream-video.mbtx run NEW_REFERENCE_ROOT
moon run scripts/run-minimax-upstream-video.mbtx inspect NEW_REFERENCE_ROOT
```

Decoded pixels, temporal stability, correspondence to the caption and actual
audio must be inspected separately. Finite arrays, clean retirement and a
successful standardized-zero reference do not complete those checks.
