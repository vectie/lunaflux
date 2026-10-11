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

Source review of the installed original denoiser confirms that CFG-distilled
FL2VA uses one positive forward per step, packed padding is a separate document,
the timestep is unscaled `1-sigma`, normalization epsilon is `1e-5`, and the
original text boundary is unnormalized `hidden_states[50]`. Native planning
already expresses these contracts. They are not demonstrated causes of the
failed picture, so this review does not introduce speculative fixes there.
Remaining encoder differences and matched original denoiser comparison remain
open; neither is waived because the two VAEs agree.

## Literal first DiT block

**Correction — 2026-10-12:** the converted-checkpoint comparison below used
the same incorrect MLP half-order assumption as native packing. It is not an
independent numerical pass. The raw FL2VA checkpoint, loaded through the
installed original converter, differs by **25.1320251% relative L2** on the
same 1,009 real rows. An exact tensor audit identifies the cause: every
non-MLP block weight matches, but converted `ff.net.0.proj.weight` is [up,gate]
and raw `mlp.fc1.weight` is [gate,up]. After exchanging the halves, all
**154,140,672** MLP values match bitwise. Both refiner MLPs have the same
bitwise-confirmed ordering. Native packing was swapping already-converted
halves, reversing gate/up throughout the denoiser and both refiners.

The shared pure packing plan now preserves converted [up,gate]; regression
tests cover block and refiner source offsets. The offline original bridge
translates converted [up,gate] to original [gate,up], without changing original
arithmetic. This is a source-layout fix, not a CUDA special case or added
request-path validation. Repaired physical replay and real media quality are
still pending. The failed previous video is retained.

Raw replay/audit root:
`/tmp/lunaflux-minimax-upstream-block-raw-quality128-20261012-v1`.
Reports: `comparison-used.txt`, `weight-layout-with-refiners.tsv`.

The first native denoising evaluation now captures its packed input, unique
time table, rotary positions, row/time mapping and document boundaries. An
offline bridge loads the same converted checkpoint tensors into the installed,
unmodified original DiT block. The omitted rotary buffer is supplied from the
exact original FL2VA table already used by native execution; it is not
recomputed. The original module runs in its real vLLM configuration and
single-rank process-group context. Setup failures are retained separately.

The successful original replay exits zero without OOM and releases CUDA owners
under a 12-GiB/no-swap limit. Root:
`/tmp/lunaflux-minimax-upstream-block-quality128-20261011-v1`.

| Literal first-block comparison | Relative L2 | Interpretation |
| --- | --- | --- |
| All 1,024 packed rows | 6.3001196% | Includes 15 alignment-padding rows |
| All 1,009 real text/audio/video rows | 0.1335998% | Matched original inputs and weights |

The original attention backend unpads the alignment rows; native execution
attends them as a separate document. That difference dominates the full-array
norm but cannot cross the document boundary into real rows. Both reports are
preserved, rather than silently discarding the padding discrepancy. Real-row
RMSE is 0.0286172 and cosine similarity is 0.999999108. This is one block at
the first step, not fifty-block/twenty-step parity or a media-quality pass.
The next comparison must trace the complete denoiser on identical noise and
conditioning; the failed native video is not repaired by this diagnostic.

The full initial-evaluation replay is now implemented: native captures each of
the fifty retired blocks, and the installed original executes each block twice,
on its literal native predecessor and its propagated original predecessor.
Only one original block's weights reside on CUDA at a time. This separates
local implementation error from accumulated drift without exceeding the
machine's unified-memory budget. Implementation is not a completed numerical
or quality result; first-step input embedding/refiner parity also remains open.

```sh
moon run scripts/run-minimax-upstream-video.mbtx prepare NEW_REFERENCE_ROOT CAPTURED_VIDEO_LATENT_F32 --latent-shape 37 8 8
moon run scripts/run-minimax-upstream-video.mbtx run NEW_REFERENCE_ROOT
moon run scripts/run-minimax-upstream-video.mbtx inspect NEW_REFERENCE_ROOT
```

Decoded pixels, temporal stability, correspondence to the caption and actual
audio must be inspected separately. Finite arrays, clean retirement and a
successful standardized-zero reference do not complete those checks.

## Repaired fifty-block replay — 2026-10-12

Committed packing fix `b97acf10` completes the native initial evaluation;
the corrected original runs both matched-native-predecessor and propagated
original-predecessor chains through all fifty blocks. Inputs, time table,
rotary table and document boundaries are literal captures, not a regenerated
noise fixture. Both runs terminate successfully with empty execution stderr.

First-block primary-document relative L2 falls from **25.1320251%** against the
raw original to **0.07388181%** after the packing repair. The final propagated
fifty-block error is **3.467995%**, versus **347.544392%** for the wrong-packed
chain. All fifty matched-input block errors are retained; the largest primary
document error is **0.2582222%** at block 49. This is a major correction, not
bitwise equality or a media-quality acceptance threshold.

Global norms are insufficient to assess the modalities separately. Literal
captured AdaLN tags identify 3 text, 414 audio and 592 video rows; 15 padding
rows remain a separate document in the preserved raw captures. At block 49:

| Modality | Matched-input block relative L2 | Propagated fifty-block relative L2 |
| --- | --- | --- |
| Video | 0.3288291% | 4.3214408% |
| Audio | 0.1030160% | 1.7358533% |
| Text | 0 | 1.3556040% |

Root: `/tmp/lunaflux-minimax-upstream-block-repaired50-20261012-v1`.
`modality-reports.tsv` retains modality attribution at seven block frontiers;
the ordinary reports retain every row of every block. The initial container
attempt failed because host-path symlinks were not visible inside its mount;
the successful retry uses hard-linked immutable weight payloads, preserving
the failed logs. Desktop collection was interrupted by disk exhaustion, then
resumed at block 29. Only reproducible row slices were removed, with recovery
ledgers; full raw tensors and numerical reports remain. The comparator can now
select primary rows directly without creating duplicate tensor files.

A fresh repaired full-media run is active at
`/tmp/lunaflux-minimax-joint-mlp-order-quality128-20261012-v1` on `.178`, using
128×128, 120 requested frames, 21 original schedule points (20 model updates),
the actual caption encoding and terminal latent capture. Its declared residency
plan is **82,132,052,256 bytes**, under the 96-GiB device limit; no competing GPU
work runs on that host. **Media quality is pending until this run terminates,
the generated video is inspected, and the audio is assessed.** Text-encoder
drift and input/refiner parity are not waived by the packing improvement.
