# MiniMax larger smoke and quality investigation — 2026-10-11

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

The exact user unit completes at **22:41:17 CST**, after **4 h 8 m 25 s**,
with exit status zero, empty stderr, no remaining CUDA owner, and a cgroup peak
of 34,053,476,352 bytes (not a complete CUDA-residency measurement). All twenty
denoising evaluations, fifty blocks, and both decoders execute. Downloaded
video is `[1,3,124,128,128]` (24,379,392 F32 bytes), and stereo audio contains
165,600 samples/channel at 32 kHz. Both arrays and both terminal latents are
finite. Buffered diagnostic logs were empty during execution; they are now
retained under `terminal/`.

**Visual quality fails:** twelve sampled frames show scrambled colored patches
and no recognizable cat. Successful execution and finite arrays are not a
semantic-quality pass. Native outputs are retained at
`/tmp/lunaflux-minimax-media-quality128-20261011-v1`; the contact sheet and WAV
are at `/tmp/lunaflux-minimax-preview-quality128-20261011-v1`. Audio has not
been independently listened to or judged for prompt correspondence.

The installed original pipeline defaults to **768×1344 and fifty steps** when
the caller supplies neither a canvas nor a step count. It accepts the smaller
explicit canvas, but that does not establish representative checkpoint quality
there. The 128×128/twenty-evaluation run is a larger smoke/attribution fixture,
not a replacement for full-resolution quality validation. No conclusion about
encoder adequacy follows from a blurred 32×32 output alone.

Future diagnostic launches use process-level line-buffered stdout/stderr, so
the existing stage prints become visible during long execution. This changes
only the offline runner's I/O; no callback, synchronization, host readback or
logging is added to production token/kernel execution. The already running
quality128 unit is left unchanged rather than losing its completed computation.

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
the installed ARM reference image. After native completion, its literal generated
latent is replayed through the original VideoVAE in
`/tmp/lunaflux-minimax-upstream-video-quality128-captured-20261011-v1`.
That run exits zero, releases its GPU, and produces the same pixel extent.
Relative L2 is **0.0428699%**, RMSE **0.000149684**, maximum pixel difference
**0.00272679**. The original contact sheet also shows scrambled colored patches.
This localizes the failed appearance upstream of the native video decoder;
it does not yet distinguish denoiser arithmetic, conditioning, or an unsupported
quality geometry. Full original-denoiser input matching is still required.
Do not substitute a zero latent or the same integer RNG seed for identical
initial noise.

The original AudioVAE also replays the literal generated audio latent, terminates
without OOM, and releases CUDA owners. It produces `[1,2,165600]`, matching the
native extent. Relative waveform L2 is **0.0982279%**, RMSE **0.0000386479**,
maximum difference **0.00150323**. Root:
`/tmp/lunaflux-minimax-upstream-audio-quality128-captured-20261011-v2`.
This is matched-latent decoder agreement, not a listening or audio-quality pass.
The incomplete `v1` weight-copy attempt was stopped and preserved; `v2` reuses
the already staged original checkpoint instead of transferring it again.

```sh
moon run scripts/run-minimax-upstream-video.mbtx prepare NEW_REFERENCE_ROOT CAPTURED_VIDEO_LATENT_F32 --latent-shape 37 8 8
moon run scripts/run-minimax-upstream-video.mbtx run NEW_REFERENCE_ROOT
moon run scripts/run-minimax-upstream-video.mbtx inspect NEW_REFERENCE_ROOT
```

Decoded pixels, temporal stability, correspondence to the caption and actual
audio must be inspected separately. Finite arrays, clean retirement and a
successful standardized-zero reference do not complete those checks.
