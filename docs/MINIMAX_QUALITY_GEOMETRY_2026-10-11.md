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

A fresh repaired full-media run executes at
`/tmp/lunaflux-minimax-joint-mlp-order-quality128-20261012-v1` on `.178`, using
128×128, 120 requested frames, 21 original schedule points (20 model updates),
the actual caption encoding and terminal latent capture. Its declared residency
plan is **82,132,052,256 bytes**, under the 96-GiB device limit; no competing GPU
work runs on that host. Its terminal and quality findings are recorded below. Text-encoder
drift and input/refiner parity are not waived by the packing improvement.

### Literal-caption conditioning replay

The installed original `ColumnParallelLinear` context projection and both
unmodified `MiniMaxH3TokenRefinerBlock` forwards now consume the **same three
BF16 caption rows** as the repaired native run. This does not substitute a
different encoder or regenerate the caption. Canonical checkpoint loading
uses the corrected converted-up/gate adapter shared with the original DiT
replay; original kernels and arithmetic remain unchanged.

Native post-refiner text rows are extracted from the literal packed initial
denoiser input. Against original final-normalized conditioning, **11,095/16,128**
BF16 values agree, global relative L2 is **0.3540210%**, and the maximum absolute
difference is **0.09375**. Individual row errors are 0.0849923%, 0.4535983% and
0.3983551%. These are not acceptance thresholds. The adapter retains context,
both block outputs and final normalization; the native prefix presently
retains only their complete final result, so this comparison does **not** yet
assign the conditioning error to an individual native operator.

Root: `/tmp/lunaflux-minimax-original-refiner-actual-20261012-v2`.
Original execution terminates zero and releases CUDA. Its stderr retains the
installed vLLM/Omni version and Gloo hostname warnings, not an empty-stderr
claim. The first preparation attempt lacked the container's output UID and
failed compilation before GPU execution; its `v1` logs are retained. The
successful bounded replay uses the existing owner UID, 8-GiB/no-swap limits
and no concurrent workload on `.179`.

```sh
moon run scripts/replay-minimax-original-refiner.mbtx prepare NEW_ROOT EXISTING_FULL_REPLAY_ROOT LITERAL_ENCODER_BF16 LITERAL_PACKED_INPUT
moon run scripts/replay-minimax-original-refiner.mbtx run NEW_ROOT
moon run scripts/replay-minimax-original-refiner.mbtx collect NEW_ROOT
```

A quiet thread continuation checks outstanding numerical/media work every ten
minutes. The repaired media run and same-latent original decoder controls are
now terminal and collected. The exact terminal collector used suffix
`mlp-order-quality128-20261012-v1`, not the full prefixed unit name. Media and
audio assessment, encoder drift and full-conditioning operator isolation remain
open. No new hardening or speculative precision change was introduced.

The shared layout adapter is independently checked by reexecuting original
block 0 on the preserved full literal input: **5,505,024/5,505,024 BF16 values
match bitwise**, including the separately retained padding document. Both
original adapters compile with `-Wall -Wextra -Werror`; runner regressions pass
(refiner 1/1, block 2/2). This verifies reference preservation, not native parity.

## Repaired media terminal and decoder controls — 2026-10-12

The repaired unit finishes at approximately **04:46 CST**, with all twenty
updates, fifty blocks per update and both decoders complete. User-unit result
is success, exit status zero, stderr empty and swap peak zero. Literal output
is **124 frames at 128×128**, not 120 decoded frames, plus stereo audio with
165,600 samples/channel at 32 kHz. Outputs and both standardized terminal
latents have no nonfinite values. Collection retains their original bytes at
`/tmp/lunaflux-minimax-mlp-order-media-output-20261012-v1`.

**Visual quality still fails after the packing repair:** the twelve-frame
contact sheet shows colored lattice-like patches, no recognizable red cat and
no intelligible depicted scene. The original VideoVAE on these exact repaired
latents shows the same failed appearance. Thus decoder implementation drift
does not explain this visual failure; conditioning, complete denoiser/head,
flow evolution and the nonrepresentative smoke geometry still need attribution.

| Original decoder on identical repaired latents | Relative L2 | RMSE | Maximum absolute difference |
| --- | --- | --- | --- |
| Video, 6,094,848 F32 pixels | 0.04290649% | 0.0002480033 | 0.010872424 |
| Audio, 331,200 F32 samples | 0.03794870% | 0.00000877134 | 0.0001439126 |

Reference roots are
`/tmp/lunaflux-minimax-upstream-video-mlp-order-captured-20261012-v1` and
`/tmp/lunaflux-minimax-upstream-audio-mlp-order-captured-20261012-v1`.
They run serially on `.179`, terminate successfully and release GPU ownership.
Their installed-module warnings are preserved; empty stderr is claimed only
for the native media run. The original waveform comparison is numerical, not
a listening, content, or audiovisual synchronization pass. Native contact
sheet/WAV are at `/tmp/lunaflux-minimax-mlp-order-media-preview-20261012-v1`.

The complete-prefix diagnostic now additionally captures the **selected final
head, literal packed pre-update states/velocities and first RF update**, with
literal inverse-time indices and sigma records. It derives the head extent
from its prepared source instead of guessing kernel positions. An independent
adapter calls the installed original final layer on literal native block-49
output and calls the original scheduler on literal native velocity. No observer,
host readback or new validation is added to ordinary serving. These comparisons
are now measured below; they do not establish full-trajectory equivalence.

The same independent replay additionally tests the original F32 video/audio
input projections and original time embedder on the captured initial states
and literal time-index ownership. This distinguishes input-producer drift from
head/flow error without reintroducing an independently generated noise fixture.

Task compatibility was checked against both the installed original pipeline
and the actual partition's `model_index.json`: the `fl2va` **checkpoint partition
supports both `t2va` and `fl2va` tasks**. This caption-only diagnostic follows
`t2va`; the original uses the verbatim caption without special tokens and no
first-frame anchors. The lack of an image is therefore not a task mismatch.
This does not establish that 128×128 and twenty updates are sufficient for
representative output quality.

The native final-head/flow diagnostic is committed and pushed as `6a29cc6e`;
its clean ARM build succeeds. First dispatch passed the campaign root instead
of its `aot` subdirectory. A preserved `openat` trace locates `ENOENT` precisely
at `video_pack.cubin`, after successful checkpoint loading—not a weight-reader
or numerical failure. Corrected run
`/tmp/lunaflux-minimax-joint-head-flow-20261012-v2` uses the actual `aot` directory;
it terminates successfully with empty native stderr and zero unit swap peak.
No existing evidence or production artifact was overwritten.

### Literal first-evaluation controls — completed

The new capture's packed input and block-49 tensor match the preserved repaired
fifty-block replay **bitwise**. Observation does not change the native body.
Installed-original replay root is
`/tmp/lunaflux-minimax-original-head-actual-20261012-v3`; local native operands
are at `/tmp/lunaflux-minimax-head-flow-capture-20261012-v2`.

| Original operator on literal native inputs | Exact values | Relative L2 | Maximum absolute difference |
| --- | --- | --- | --- |
| Final video head, native block 49 | 1,506 / 56,832 F32 | 0.0001438434% | 0.0000467300415 |
| Final audio head, native block 49 | 351 / 13,248 F32 | 0.0001272711% | 0.0000129938126 |
| First video RF update, native state/velocity | 56,832 / 56,832 F32 | 0, bitwise | 0 |
| First audio RF update, native state/velocity | 13,248 / 13,248 F32 | 0, bitwise | 0 |
| Time embedder, actual unique times | 3 / 2,688 F32 | 0.0015086934% | 0.00000543892384 |
| Video input projection, native initial state | 3,182,147 / 3,182,592 BF16 | 0.0024798051% | 0.0625 |
| Audio input projection, native initial state | 2,225,437 / 2,225,664 BF16 | 0.0009477162% | 0.0078125 |

These controls exclude a large **first-evaluation** input/head/update error.
They neither waive accumulated body/conditioning error nor prove representative
quality. Text and padding inherit **video time**, audio inherits audio time.
The diagnostic's initial-time reconstruction was corrected accordingly in
`3f8f7321`, with compile-time cases for unequal times; production already used
that convention. The installed original warnings remain preserved.

### Full original trajectory control — completed, original smoke also fails

`scripts/replay-minimax-original-trajectory.mbtx` calls the installed original
`MiniMaxH3DiTModel`, packed-sequence builder and denoise loop. It receives the
same native initial F32 noise, actual three-row encoded caption, literal
twenty-update sigma schedule, original checkpoint and rotary table. It compares
packed positions, document boundaries and initial modality ownership before
execution, records all twenty post-update states, and unpacks terminal latents
using installed original helpers. The independently generated original sigma
routine is also reported; the integration comparison uses captured native
points rather than silently changing its inputs.

The model is instantiated **directly on CUDA** to avoid a second full CPU
weight copy. Checkpoint tensors are supplied one at a time. Neither the 51-GB
encoder nor either VAE coexists with the approximately 66-GB DiT. Admission
requires an idle `.179` GPU and at least 104 GiB available memory, with a
96-GiB process/container ceiling and zero permitted unit swap. Decoding runs
only after the original DiT releases the GPU. This is offline comparison setup,
not new request-path validation or production hardening.

The full installed-original trajectory completes at approximately 05:34 CST,
exit zero, all twenty updates, and releases the GPU. Original warnings are
retained. Root:
`/tmp/lunaflux-minimax-original-trajectory-quality128-20261012-v2`.
Packed F32 positions, document boundaries and initial modality ownership all
match exactly. The reference consumes the literal native sigma points; the
original independently generated 21-point schedule differs by at most
2.98023224e-8 at one video point and two audio points, explicitly reported
rather than called bitwise-equivalent. Its original time/flow routines remain
unchanged.

| Complete original trajectory, matched noise/caption/geometry | Relative L2 | Maximum absolute difference |
| --- | --- | --- |
| First complete video update versus native | 0.01948708% | 0.0009775385 |
| First complete audio update versus native | 0.01673137% | 0.0006986260 |
| Terminal standardized video latent versus native | 2.69715095% | 0.60274291 |
| Terminal standardized audio latent versus native | 1.62026389% | 0.13460186 |

All compared values are finite. The first-update differences include the
complete producer/body/head chain; they are not the isolated head or scheduler
errors above. Terminal errors accumulate over repeated nonlinear evaluations.
These are measured differences, not acceptance thresholds or quality claims.
All twenty original updated video/audio states are retained locally without
duplicating model weights. The original decoder is now testing these **original
trajectory latents**, not merely decoding native latents again.

The initial trajectory preparation lacked a desktop rotary-table copy and
stopped before compile/GPU execution. Its incomplete `v1` directory is preserved;
`v2` copies the exact already-retained host rotary buffer. The first original
trajectory video preparation also stopped during management-source copying
before GPU admission. The runner can now reuse the already retained unchanged
reference source in a fresh directory; this avoids unrelated source re-fetching
without changing decoder arithmetic.

The installed original VideoVAE subsequently decodes these **original** terminal
latents in
`/tmp/lunaflux-minimax-upstream-video-original-trajectory-20261012-v2`, exits
zero and releases CUDA. Literal output remains `[1,3,124,128,128]`. The twelve
sampled frames were actually inspected at
`/tmp/lunaflux-minimax-original-trajectory-preview-20261012-v1/frames.png`:
they also show colored lattice-like patterns and no recognizable red cat.
Thus the native-only body is not sufficient to explain the failed smoke
appearance. This original control retains the identical supplied **native
encoder rows**, not an independently re-encoded caption; it cannot yet exclude
the shared caption input or establish which larger geometry succeeds. Neither
run earns a media-quality pass.

The native offline prefix capture now additionally retains context projection,
each actual refiner boundary and final conditioning. Stage extents are derived
from the unchanged bound layer launch descriptions, not guessed kernel indices.
Ordinary text conditioning still uses its original single queue/completion with
no intermediate copy or observer branch. These finer controls are implemented
but must be physically measured before assigning the remaining 0.354% complete
conditioning difference to any particular operator.

No new production arithmetic or precision policy is selected by these findings.

### Retired conditioning frontiers — matched local operators

Clean main `1d0a0ed6` passes the full native suite **5,021/5,021** and ARM
compilation. Its `.178` prefix capture completes in
`/tmp/lunaflux-minimax-joint-refiner-stages-quality128-20261012-v1`, exit zero,
empty native stderr, with CUDA released. Final conditioning is **16,128/16,128
bitwise-identical BF16 values** to the preceding native capture: intermediate
observation does not change the arithmetic.

The original refiner replay in
`/tmp/lunaflux-minimax-original-refiner-stages-20261012-v3` supplies each literal
native predecessor independently, rather than propagating a different original
output and misattributing accumulated error to the next operator.

| Local original stage on identical native predecessor | Exact BF16 / 16,128 | Relative L2 |
| --- | --- | --- |
| Context projection on actual encoded caption | 16,076 | 0.02407941% |
| Refiner block 0 on native context | 12,395 | 0.05489051% |
| Refiner block 1 on native block 0 | 14,075 | 0.03818560% |
| Final RMSNorm on native block 1 | 16,128 | 0, bitwise |

All values are finite. Complete conditioning's earlier 0.354021% difference
includes propagation through the nonlinear blocks; it is not a demonstrated
final-normalization bug. Local block-0 row-relative errors reach 0.21775% and
block-1 0.21080%; the global norm is dominated by the large first caption row.
Original warnings are retained. The reported 11.4-MB user-unit memory peak is
the Docker client measurement, **not** complete CUDA/container residency.
No precision change or correctness acceptance follows merely from these sizes.

### Original encoder caption control — original smoke still fails

`/tmp/lunaflux-minimax-original-trajectory-original-caption-20261012-v3` replaces
the supplied native encoder rows with the installed original encoder's actual
three rows for the same caption. It retains native initial noise, literal
21-point schedule, checkpoint, geometry and original denoiser/packing/flow.
All twenty updates complete and release CUDA. This is explicitly a
**changed-caption-input control**, not an identical-caption local comparison.
Versus native terminal latents, video/audio relative L2 are
**3.30244827% / 3.43541439%**; all finite, not quality acceptance.

The original VideoVAE subsequently completes at
`/tmp/lunaflux-minimax-upstream-video-original-caption-20261012-v1` with actual
output `[1,3,124,128,128]`. Its twelve sampled frames were actually inspected at
`/tmp/lunaflux-minimax-original-caption-preview-20261012-v1/frames.png`:
colored lattice/no recognizable red cat persists. Native caption encoding is
therefore not the sole cause of this failed smoke appearance. This does not
prove that larger geometry succeeds or that every installed reference adapter
is a validated representative-quality pipeline.

Next schedule-only control uses the original pipeline's default **50 sigma
points / 49 updates**, same original encoder caption/noise/128 geometry. Root
`/tmp/lunaflux-minimax-original-trajectory-original-caption50-20261012-v1`
completed all 49 updates serially on `.179` under the same bounded-memory setup. Changed
schedule is labeled explicitly and cannot earn matched twenty-update parity.
Its larger step count does not make 128×128 representative-quality evidence.
Audio remains quantitatively compared but not listened to; no audio-quality
pass is claimed.

Its installed-original VideoVAE completes in
`/tmp/lunaflux-minimax-upstream-video-original-caption50-20261012-v1`.
Actual geometry remains `[1,3,124,128,128]`. The contact sheet at
`/tmp/lunaflux-minimax-original-caption50-preview-20261012-v1/frames.png` was
inspected: lattice-like color patterns persist, no recognizable cat. The
default step count does not rescue this small-geometry result. Next quality
work must check shared fixture/reference setup and a representative positive
original control, not claim that native rounding alone caused this appearance.

The next native diagnostic derives named norm/QKV/QK-norm/attention/projection,
residual and gate/up/down frontiers from the **same immutable refiner launch
description** used by ordinary binding. Scratch ownership/layout remains in
`engine/joint_diffusion_device`; offline readback consumes a retired stage and
does not enter ordinary execution. This adds operator-level attribution rather
than another guessed precision change. Warning-denied native checking and
affected tests pass **55/55**, full native suite **5,022/5,022**; physical
operator measurements remain pending.

### Operator observation geometry correction

The first original operator replay at
`/tmp/lunaflux-minimax-original-refiner-operators-20261012-v1` stops after
norm1 with `component extent differs`. Inspection identifies a new diagnostic
bug: the native observer sized planar Q/K/V and attention output using hidden
width 5,376, but this checkpoint has **56 heads × 128 = 7,168** attention
channels. Production launch binding already uses the correct head geometry;
this is not evidence of a serving QKV geometry defect.

The corrected observer derives byte ranges from the prepared workspace's
head count and head dimension. For three text rows, planar Q/K/V requires
129,024 bytes and attention output 43,008 bytes, whereas normalized/residual
state remains 32,256 bytes. A regression uses unequal hidden and attention
widths. The original adapter reads head geometry from its installed module,
not an assumed `hidden/head_dimension` ratio. Failed original output and
truncated diagnostic captures are preserved; fresh captures and comparisons
are required before interpreting any operator-level result.

Even the first observer's final conditioning is bitwise unchanged: all
16,128 BF16 values match the earlier native capture. That only proves observing
did not change execution, not that its scratch readback ranges were correct.

The corrected committed observer `2fcd361f` completes with empty stderr in
`/tmp/lunaflux-minimax-joint-refiner-operators-width-quality128-20261012-v1`.
All corrected scratch extents are retained locally in
`/tmp/lunaflux-minimax-refiner-operators-width-native-20261012-v1`; final
conditioning remains bitwise identical to the prior native result. Full native
suite passes **5,023/5,023**. Original operator replay `...operators-20261012-v2`
completes serially on `.179`, releases CUDA and retains its installed warnings.

Every table entry below supplies the **native predecessor independently**.
`refiner-0.bf16`/`refiner-1.bf16` in this intervention are just final residual
adds on native norm2/MLP frontiers, not whole original refiner blocks.

| Local operation | Layer 0 relative L2 | Layer 1 relative L2 |
| --- | --- | --- |
| Norm1 | bitwise | 0.00413304% (one BF16 value) |
| QKV projection | 0.02390364% | 0.01584161% |
| QK norm | bitwise | 0.00101676% (one BF16 value) |
| Attention | 0.05662943% | 0.09536291% |
| Output projection | 0.01785792% | 0.02357913% |
| Attention residual add | bitwise | bitwise |
| Norm2 | bitwise | bitwise |
| Gate/up activation | 0.00862203% | 0.00142271% |
| Down projection | 0.02446209% | 0.03988011% |
| Final MLP residual add | bitwise | bitwise |
| Final conditioning RMSNorm | bitwise | bitwise |

### Attention's remaining difference is a specific rounding contract

The independent MoonBit F64 oracle reads each layer's **literal native planar
Q/K/V** and evaluates complete noncausal three-row softmax. It does not call
either CUDA backend or re-encode the caption. Native attention matches its
BF16 output in **21,503/21,504** and **21,502/21,504** values; relative L2 is
0.000005536% / 0.000574302%. The installed original backend differs from this
F64 result by 0.05662997% / 0.09536477%.

An explicit alternative rounds **unnormalized** `exp(score-global_max)` to
BF16 before P×V, retaining the original unrounded denominator. On these
identical operands, that oracle reproduces the installed original attention
**bitwise: 21,504/21,504 in both layers**. Rounding normalized probabilities
instead does not reproduce it. Thus blanket precision widening is not the
reference-matching fix: native's existing F32 weights are more accurate to this
oracle, but implement a different probability-storage law.

`compiler/precision_ir` now expresses `GlobalUnnormalizedBf16` alongside the
existing global/online laws. The staged joint CUDA renderer can lower it while
keeping F32 maxima/denominator/accumulation. Unsupported online/global
combinations are explicit, not silent substitutions. Existing production
selection remains **GlobalF32** pending exact-source CUDA, sanitizer and full
conditioning/trajectory measurements. This is an arithmetic-contract candidate,
not a tolerance waiver or a new runtime dependency. CPU oracle root:
`/tmp/lunaflux-minimax-refiner-attention-oracle-20261012-v1`.

The shared literal initial-noise fixture also has no obvious scale/correlation
defect: video 56,832 samples have mean 0.0038243, variance 1.0027756; audio
13,248 samples mean -0.0118228, variance 0.9994523. Absolute tested lag
correlations (1,2,32,96) are below 0.008. These are distribution diagnostics,
not equivalence to the original RNG, a model-input quality pass, or proof that
the failed 128 geometry is the sole cause. Installed t2va presentation explicitly
uses verbatim prompt tokens without a chat template; adding a guessed system
prompt is not justified by its source.

### Exact-source CUDA probability intervention

The fresh bounded physical probe at
`/tmp/lunaflux-minimax-refiner-probability-20261012-v1` replays the literal
captured Q/K/V for both layers. Its unchanged F32 source reproduces the native
capture bitwise, confirming that this intervention tests the executed source.

| Identical-input attention | Existing F32 law | Unnormalized BF16 law |
| --- | --- | --- |
| Layer 0 exact original values | 20,991 / 21,504 | **21,504 / 21,504** |
| Layer 1 exact original values | 19,511 / 21,504 | **21,503 / 21,504** |
| Layer 1 relative L2 | 0.09536291% | **0.00459427%** |

The remaining layer-1 value differs by 0.03125; it is retained and reported,
not waived. Memcheck plus full leak checking report zero errors/zero bytes
leaked, racecheck zero hazards, synccheck zero errors. CUDA resources are
released between serial checks. This proves a local reference-matching
improvement, not full encoder/conditioning parity or a media-quality pass.

The refiner builder now accepts the same explicit generic probability law;
the default remains F32. Its regression verifies that selecting the BF16 law
changes only the numerator's probability conversion, preserving every other
generated operation and symbol. A separate whole-conditioning AOT experiment
uses that source without replacing the original bundle or changing production
selection. Failed visual controls and unresolved DeepSeek results remain
unchanged by this isolated experiment.

Whole conditioning now completes with this single changed probability law in
`/tmp/lunaflux-minimax-joint-refiner-probability-quality128-20261012-v1`.
It exits zero, empty stderr, no unit swap; captured operators and conditioning
are preserved under `/tmp/lunaflux-minimax-refiner-probability-native-20261012-v1`.
Against the same original full conditioning/encoded caption, **relative L2 is
0.35584173% versus baseline 0.35402103%**, and exact BF16 values are
11,030 versus 11,095 out of 16,128. Thus the local attention improvement does
**not** improve this whole-chain comparison; production remains unchanged.
This negative result is not silently relabeled as a completed encoder fix.
An original propagated-refiner replay beginning at the literal native context
projection will isolate projection drift from subsequent original arithmetic;
unlike per-block interventions, its second block consumes its own first output.

That control completes at
`/tmp/lunaflux-minimax-original-refiner-native-context-20261012-v1`, using the
installed original two refiners and final norm with the **same initial native
context** as both native alternatives. Context is bitwise unchanged. Baseline
final conditioning differs by **0.34671401%**; the BF16-probability candidate
differs by **0.29873496%** (11,978 versus 10,997 exact values). Thus the
refiner-only propagated comparison improves about 13.8% in relative-error norm,
while the complete original-projection-plus-refiner comparison above does not.
These are different interventions; neither replaces the other. The candidate
has a genuine local and same-context-chain benefit, not a demonstrated complete
conditioning or media-quality fix. Original projection and remaining refiner
GEMM/reduction/storage behavior still require independent attribution.

Full native suite passes **5,026/5,026**, affected compiler/source packages
**64/64**, and offline refiner/oracle/noise/bundle regressions pass. Existing
toolchain migration-warning exclusions are unchanged. The refiner adapter
continues calling installed original operations; new context-chain mode changes
only its supplied initial input and propagates original outputs thereafter.

### Bounded original RNG / larger-canvas control

The three failed 128×128 original controls still used captured native initial
noise. Its measured marginal statistics look normal, but that is not an
independent check of the installed pipeline's noise/patchifier route. A new
explicit **original-only quality control** calls the installed
`MiniMaxH3Pipeline._initial_noise` unchanged, with seed 17, the actual original
three-row caption, 256×256 canvas and fifty original sigma points / 49 updates.
It uses the same installed original full DiT and checkpoint weights; the task
remains text-to-video/audio. Neither a seed nor a shared caption asserts
identical noise after the canvas changes.

Root / exact user unit on `.179`:
`/tmp/lunaflux-minimax-original-trajectory-original-rng-quality256-20261012-v1`
/ `lunaflux-minimax-original-trajectory-original-rng-quality256-20261012-v1.service`.
The original DiT completes all **49 updates**, exits zero, releases CUDA and
has `MemorySwapPeak=0`. It has a 96-GiB
resident-memory limit, zero unit swap and a two-hour timeout; no other GPU
work is admitted there. Video has 2,368 patch rows, audio 414, caption 3;
the canonical terminal video latent is `[1,24,37,16,16]`. Actual decoded
geometry and semantic content are checked after the original DiT releases
the GPU and the original VAE runs separately. Original VideoVAE root
`/tmp/lunaflux-minimax-upstream-video-original-rng256-20261012-v1` also exits
zero, releases CUDA and reports zero unit swap. Its literal output is
**`[1,3,124,256,256]`, 97,517,568 F32 bytes**. Installed original warnings remain
in the logs; stderr is not relabeled empty.

The twelve-frame contact sheet now shows a **recognizable reddish-brown cat**,
unlike the failed 128×128 controls. However, it also shows conspicuous lattice
texture, malformed/glowing eye features and changing saturated red/green/blue
patches. This establishes prompt-related visual content at this control, not
a clean media-quality pass or full-resolution validation. Original geometry
and initial noise both changed, so their individual effects are not identified.
Preview: `/tmp/lunaflux-minimax-original-rng256-preview-20261012-v1/frames.png`.
Audio is not decoded or listened to in this new control.

A separate original-noise **128×128**, same-caption / 49-update control completes
at
`/tmp/lunaflux-minimax-original-trajectory-original-rng-quality128-20261012-v1`.
It and the separate original decoder
`/tmp/lunaflux-minimax-upstream-video-original-rng128-20261012-v1` exit zero,
release CUDA and record zero unit swap. Literal decoded output remains
`[1,3,124,128,128]`. Its inspected contact sheet again shows scrambled colored
lattice and **no recognizable cat**:
`/tmp/lunaflux-minimax-original-rng128-preview-20261012-v1/frames.png`.
This restores the old canvas while using the installed original noise route.
The 128 failure is therefore not unique to LunaFlux's random generator. The
256 control provides recognizable prompt content, but neither original control
is a clean quality pass; changing canvas also changes actual sampled input
positions, so the experiment does not identify geometry as the sole cause.
It is not a native numerical parity replay or a production RNG selection change.

The offline bridge now makes observer extents and unpatchification geometry
explicit rather than retaining hard-coded 592-row / 8×8 assumptions. Existing
matched 128×128 replays keep their literal captured noise, packing identity
checks and output extents. Changed-noise/geometry runs cannot enter the native
parity collector: `collect-quality` downloads only original initial operands
and terminal latents, retaining all step captures on the host. The runner's
three regressions and ARM bridge compilation with `-Wall -Wextra -Werror` pass.
This is diagnostic plumbing, not a production arithmetic/default change or
proof that 256×256 is representative full-resolution checkpoint quality.

### Literal-original initialization for the next native replay

The native checkpoint diagnostic now has explicit `run-joint-*-literal` modes.
They read the two complete original initial tensors before opening CUDA, using
the prepared latent plan for their extents, then upload them once into the
existing owned latent arena. Canonical NCTHW video still passes through the
ordinary native patchifier; original token-major audio is copied unchanged.
The seed is explicitly not used in this mode. Normal seeded initialization,
model semantics, kernels and serving defaults are unchanged; no filesystem or
reference operation is added to denoising steps.

`scripts/unpatch-minimax-original-noise.mbtx` performs only an exact byte
permutation from original `[T,H/2,W/2,C,2,2]` video rows to canonical `[C,T,H,W]`,
retaining original F32 bits and original audio rows. Regressions cover spatial,
channel and time ownership. `scripts/run-minimax-joint.mbtx` accepts a final,
explicit `--initial-noise DIRECTORY` control, preserving separate prefix,
full-prefix and terminal-latent modes. This closes an input-matching gap; it
does not repair encoder arithmetic or establish media quality. A fresh native
256×256 prefix must verify its captured initial rows against original rows
before any downstream numerical comparison is interpreted.

Committed native implementation `bb617c57` builds cleanly on ARM and passes
the local **5,027/5,027** native suite. The 256×256/50-point AOT export completes,
reusing 23 identical modules and compiling 8 changed-geometry modules in a new
directory. The first launch
`/tmp/lunaflux-minimax-joint-original-noise256-prefix-20261012-v1` fails **before
CUDA setup**: diagnostic metadata newlines entered the command, truncating its
argument list. That root/unit and logs are preserved; it is not a numerical or
memory-pressure failure. The launcher now isolates initial-output arguments
from metadata and passes a regression rejecting embedded metadata lines.
The revised runner has **10/10** tests; original-noise unpatchification has
**1/1**. A fresh `v2` prefix will reuse the same committed native executable,
original operands and newly compiled AOT modules without overwriting `v1`.

The corrected `v2` native prefix exits zero with empty stderr, zero unit swap
and one completed DiT block. Its planned resident total is **82,890,073,512
bytes**, below the existing 96-GiB budget; literal startup mode is recorded and
the seed is not used. This is not a terminal media run. The diagnostic prefix
now also retains actual initial video/audio rows after native patchification,
even when stopping after one block, so a fresh capture can demonstrate byte
identity with original rows rather than relying on supplied filenames alone.

The original block adapter's old 1,024 packed / 1,009 real-row constants are
removed. It derives packed row count, primary document and maximum document
length from the actual captured boundaries, within its existing bounded
single-block residency strategy. At 256×256 this is **2,816 packed / 2,785 real
rows**, not the old smoke geometry. The original module's arithmetic is
unchanged. Its runner regressions cover both canvases, and the larger adapter
compiles in the installed ARM image with `-Wall -Wextra -Werror`.

### Matched 256 first-block probability intervention

The `v3` native prefix completes with empty stderr and released CUDA. After
the ordinary native patchifier, all **227,328 video F32 values** and **13,248
audio F32 values** are bitwise identical to installed-original startup rows.
The installed original block then consumes the literal native packed input,
time table, positions, tags and documents, at 2,816 packed / 2,785 real rows.
Its unchanged first block completes under the bounded single-block budget.
Original installed warnings remain present in stderr.

A separate candidate changes only `dit_n1`'s attention probability numerator
storage to `GlobalUnnormalizedBf16`. Its generated baseline reproduces the
executed baseline translation unit byte-for-byte. All other AOT modules,
including both text refiners, are unchanged. The native input is bitwise equal
to the baseline prefix, and both initial noise arrays again match the original.
Root: `/tmp/lunaflux-minimax-joint-block-probability256-prefix-20261012-v1`.
It completes one block, not a full trajectory or terminal media run.

| Real-row comparison with the same original block output | Default F32 P | BF16 unnormalized P candidate |
| --- | ---: | ---: |
| Combined relative L2 | 0.0841358% | 0.0711123% |
| Video relative L2 | 0.2891593% | 0.2325272% |
| Audio relative L2 | 0.1004746% | 0.0818648% |
| Text relative L2 | 0.0418689% | 0.0405842% |
| Exact real-row BF16 values | 11,502,428 / 14,972,160 | 12,050,986 / 14,972,160 |

Probability storage explains part of the local difference, not all of it.
The combined norm is dominated by large text features; modality attribution
is retained rather than using that aggregate as a quality claim. This is not
a full-stack, all-timestep, encoder, performance or media-quality pass.
The production/default probability law remains F32. The source builder merely
forwards the existing explicit numeric contract; unchanged timestep/output
source and byte-identical baseline source have regression coverage.

The audio inspection tool in this conversation does not support audio input.
The WAV is retained, but listening quality has **not** been assessed here;
finite arrays and close decoder outputs cannot substitute for that assessment.

The next native prefix observer retires the first block's actual bound stream
at named norm/affine, QKV, QK-normalization/RoPE, attention, projection and MLP
frontiers, before the later MLP reuses attention scratch. It never records a
whole-executor completion event at an intermediate frontier or replays input
producers. The normal serving path remains observer-free. An offline original
adapter consumes each captured native predecessor independently for attention,
output projection, gate/up and down. This distinguishes local arithmetic from
propagated conditioning/block differences; results are pending, not a new
correctness claim. All captures stay in separate non-overwriting roots.

This first-block capture now completes at
`/tmp/lunaflux-minimax-joint-block-frontiers256-prefix-20261012-v1`, with empty
stderr and released CUDA. Its input and final block output are bitwise equal
to the earlier BF16-P candidate, demonstrating unchanged arithmetic despite
the intermediate observation waits. Literal original startup rows again match.
The committed native suite passes **5,031/5,031**.

Installed original local operators also complete at
`/tmp/lunaflux-minimax-upstream-block-operators256-20261012-v1`, using each
native predecessor separately, not a propagated reference chain. The original
block weights, cuDNN attention backend and separate BF16 SiLU/multiply are
unchanged. Actual real-row differences are:

| Matched local operation | Video relative L2 | Text relative L2 | Audio relative L2 |
| --- | ---: | ---: | ---: |
| Attention, candidate BF16 P | 0.0986211% | 0.0542969% | 0.0984160% |
| Output projection | 0.0235109% | 0.0124977% | 0.0201437% |
| Gate/up | 0.0285391% | 0.0100834% | 0.0285161% |
| Down | 0.0272605% | 0.0200258% | 0.0271991% |

These isolate attention as the largest of these four local differences for
this block/input; they do not prove it is the largest source across all layers
or denoising timesteps. Norm, modulation, QKV/RoPE and conditioning still need
their own matched attribution. No media-quality pass follows from this table.

The exact candidate translation unit is independently replayed for its **first
attention CTA only** on captured Q/K/V. All 256 BF16 outputs reproduce the
native capture in 10 repeats; ordinary CUDA-event mean is **12.2842 ms** for
this partial work domain (not whole-kernel or serving performance). Allocations
and events are explicitly released. Memcheck/full leak check reports zero
errors/leaks, racecheck zero hazards/errors/warnings, synccheck zero errors.
Root: `/tmp/lunaflux-minimax-block-frontier-probability256-20261012-v1`.
These sanitizer observations do **not** cover the whole 2,816-row work domain.
The production/default probability law remains unchanged.

An independent F64 oracle now evaluates the first, middle and last real query
of **each modality**, at heads 0, 1, 28 and 55, retaining the entire 2,785-key
document for every query. It compares 36 query/head pairs / **4,608 BF16 output
values**. Padding is not used to rank the alternatives.

| Oracle probability storage / partition | Exact original outputs / 4,608 |
| --- | ---: |
| Unrounded F64 mathematical softmax | 3,818 |
| Global BF16 unnormalized P | 4,054 |
| Online BF16 P, 32 keys | 4,470 |
| Online BF16 P, 64 keys | **4,598** |
| Online BF16 P, 128 keys | 4,348 |
| Online BF16 P, 256 keys | 4,228 |
| Online BF16 P, 512 keys | 4,127 |

The global BF16 oracle also reproduces **4,604/4,608** native candidate values.
This identifies a strong rounding/partition explanation for the residual
attention difference on this subset; it does not prove the original backend's
tile geometry from SASS or establish other inputs. The oracle uses F64 dot,
rescaling, denominator and PV accumulation, unlike production F32 arithmetic.
File: `subset-attention-laws.tsv` in the matched-operator root above.

The joint CUDA renderer now has an explicit lowering for the already existing
`OnlineBf16(block_tokens=...)` precision-IR contract. Its partition remains a
validated parameter, not a MiniMax-specific implicit constant. It rescales
old numerator/denominator mass, rounds only the tile P operand and handles
document tails without integer-overflow-prone loop increments. The previous
global translation unit remains byte-identical, including the default route.
The candidate's numerical contract records the partition. A fresh 64-key
first-block experiment now completes; it is not production-selected or a
full-model/media correctness result. No reference arithmetic or tolerance
is changed to make that experiment pass.

Root: `/tmp/lunaflux-minimax-joint-block-online64-256-prefix-20261012-v1`.
Packed input and the complete normalized/rotated QKV operands are bitwise
equal to the earlier candidate. Initial original noise matches again; stderr
is empty and CUDA owners retire. Compared with the same installed-original
attention output on these identical operands:

| Modality | Global BF16 P local attention error | Online BF16 P64 local attention error |
| --- | ---: | ---: |
| Video | 0.0986211% | **0.0222604%** |
| Text | 0.0542969% | **0.0138850%** |
| Audio | 0.0984160% | **0.0228316%** |

These are **numerical error reductions**, not throughput or latency speedups.
Whole-block video error moves from 0.2325272% to **0.2089222%**, text from
0.0405842% to **0.0393180%**, but audio moves from 0.0818648% to **0.0823538%**.
All three remain better than the original default-F32 first-block comparison,
but online64 does not improve every modality versus the intermediate global
BF16 experiment. Local reductions cannot be added or converted directly into
whole-block gains: upstream differences and downstream residual arithmetic
can amplify or cancel. Norm/modulation, QKV/RoPE and complete trajectory
correctness remain unresolved.

The exact online64 translation unit's first-CTA probe reproduces 256/256
captured BF16 outputs in ten repeats; memcheck/full leak, racecheck and
synccheck report zero errors/leaks/hazards. Ordinary event mean is 11.3831 ms
for that one CTA, not the entire attention operation or serving workload.
The same partial-domain sanitizer limitation applies as before. Probe root:
`/tmp/lunaflux-minimax-block-frontier-online64-256-20261012-v1`.

Immutable block/refiner source values now retain their explicit probability
law, instead of discarding it after rendering source. Prediction-variant
preflight rejects inconsistent laws between timestep buckets, so a variant
selection cannot silently change precision semantics. This is a one-time
semantic plan join, not filesystem validation or request-path qualification.
Both default laws are unchanged. The native suite passes **5,033/5,033**.

The original256 terminal audio latent also decodes separately through the
installed original AudioVAE at
`/tmp/lunaflux-minimax-upstream-audio-original-rng256-20261012-v1`. Literal
output is `[1,2,165600]` F32, 1,324,800 bytes, **5.175 seconds at 32 kHz**.
Execution exits zero and releases CUDA; installed warnings are retained.
Objective signal statistics are finite, RMS 0.1920855, peak 1.0, zero values
outside the PCM range. The audio-only renderer preserves a WAV for listening,
but these metrics establish neither semantic content nor listening quality.
Preview: `/tmp/lunaflux-minimax-original-rng256-audio-preview-20261012-v1/audio.wav`.

### Remaining first-block frontiers

The prefix diagnostic now also retains all six AdaLN fields for every actual
unique timestep/modality, immediately after the parameter producer retires,
and the complete attention residual before norm2. These are observer-only
changes: normal serving still has no readbacks, extra waits or host buffers.
The installed-original adapter can now isolate norm1/norm2, parameter
projection, both affine modulations, QKV, QK norm/RoPE and both gated residuals
on their literal native predecessors. No original operation is reimplemented
inside the adapter, and local oracle outputs are not fed into later operations.

The modality comparator explicitly handles planar Q/K/V ownership, retaining
all three planes and excluding only each plane's declared padding rows. Large
input and oracle tensors transfer over the existing cluster link; CPU reduction
runs on the cluster under a 2-GiB/no-swap limit and returns only small reports.
This avoids consuming the desktop's remaining disk space or overlapping GPU
work. New front-operator numerical results are pending, not a correctness fix
or complete trajectory/media-quality claim.

### Matched first-block front results — 2026-10-12

The original front replay and native capture have now completed independently,
with the original operation receiving its literal native predecessor at each
frontier. CPU comparisons cover all 2,785 real rows: video 2,368, text 3,
audio 414. Q and K are also reduced separately, so unchanged V does not hide
normalization/rotary differences.

| Local operation | Video relative L2 | Text relative L2 | Audio relative L2 |
| --- | ---: | ---: | ---: |
| Norm1 | 0.00093691% | bitwise | 0.00117152% |
| Attention affine modulation | bitwise | bitwise | bitwise |
| QKV projection | 0.01752048% | 0.01466033% | 0.01682853% |
| Q normalization/rotary, without V | 0.00275094% | bitwise | 0.00201013% |
| K normalization/rotary, without V | 0.00323687% | bitwise | 0.00132681% |
| Norm2 | 0.00105047% | bitwise | 0.00058296% |
| MLP affine modulation | bitwise | bitwise | bitwise |
| Attention gated residual | bitwise | bitwise | bitwise |
| MLP gated residual | bitwise | bitwise | bitwise |

Original AdaLN projection on the literal native F32 timestep embedding matches
**96,762/96,768 BF16 values**, relative L2 **0.002914699%**, maximum difference
0.00390625. These parameters have six fields, three modalities and 5,376
channels at this unique timestep. All compared values are finite.

This excludes a large first-block parameter permutation, affine rounding or
gated-residual formula mismatch on this input. It does not establish other
blocks/timesteps, encoder parity or media quality. Remaining local projection
and attention discrepancies propagate nonlinearly; their norms are not an
additive whole-chain error budget. The ordinary production arithmetic and
probability selection remain unchanged.

Roots:

- Native: `/tmp/lunaflux-minimax-joint-front-operands256-prefix-20261012-v1`.
- Original: `/tmp/lunaflux-minimax-upstream-block-front-operators256-20261012-v1`.
- CPU reports: `/tmp/lunaflux-minimax-front-comparison-256-20261012-v3`.
- Small parameter comparison: `parameters-comparison.txt` in comparison `v1`.

The diagnostic CPU runner now uses the native target and a task-local SDK copy
without cloning the old SDK's incompatible dependency cache. Earlier Wasm/cache
failures are retained; they occurred before numerical comparison, not in model
execution. Large arrays remain on the cluster, and report units use a 2-GiB,
zero-swap limit. No GPU work overlaps these CPU reductions. Original installed
warnings remain recorded; native prefix stderr is empty.

### Online64 complete-stack intervention in progress

The next isolated experiment runs the same 256×256, original-encoder caption,
literal original initialization and first timestep through **all 50 denoiser
blocks**, followed by the actual output heads and first Euler update. It does
not run a full denoising trajectory or decode media. Only the shared DiT
probability source is the unselected online64 candidate; refiner, projection,
modulation, residual, head and flow sources stay unchanged.

Native root `/tmp/lunaflux-minimax-joint-online64-256-full-prefix-20261012-v1`,
exact user unit
`lunaflux-minimax-joint-online64-256-full-prefix-20261012-v1.service`, started
2026-10-12 08:36:04 CST. Source-pinned executable is from `f50b4ce3`; the AOT
is `/tmp/lunaflux-minimax-request-aot-block-online64-256-20261012-v2/aot`.
The planned weight/workspace/reserve total is 82,890,073,512 bytes, below the
96-GiB device ceiling. The user unit has a 96-GiB memory cap, zero swap and
7,200-second runtime cap. Process-cgroup memory is not total GPU residency.
Do not overlap another GPU operation on `.178` while this unit is running.

Original comparison root is prepared separately at
`/tmp/lunaflux-minimax-upstream-block-online64-256-full-20261012-v1` on `.179`.
Once native is successfully terminal, `inputs-cluster-full` transfers the
literal packed input and all 50 native block outputs directly between cluster
nodes. `run-full` then produces independent matched-input and propagated
original series under the existing 12-GiB/no-swap limit. Cluster CPU reduction
returns modality reports for blocks 0/1/10/20/30/40/49 without downloading the
multi-gigabyte tensor series to the desktop. No result or quality pass is yet
claimed for this in-progress complete-stack intervention.

The native complete-stack intervention is now terminal: all 50 blocks, the
output heads and first Euler update complete, exit zero with empty stderr and
released CUDA owners. The installed-original full replay also completes on
`.179`, using the same packed input/timestep/positions and each literal native
predecessor independently, plus a separately propagated original series.
Cluster-only report root:
`/tmp/lunaflux-minimax-front-comparison-online64-full-20261012-v1`.

| Block | Local matched video error | Propagated video error | Propagated text error | Propagated audio error |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.208922% | 0.208922% | 0.039318% | 0.082354% |
| 1 | 0.165514% | 0.268262% | 0.081756% | 0.074053% |
| 10 | 0.028449% | 0.379610% | 0.118381% | 0.197370% |
| 20 | 0.062621% | 1.065413% | 0.610106% | 0.215064% |
| 30 | 0.211125% | 2.566078% | 0.448590% | 0.446194% |
| 40 | 0.154002% | 5.597277% | 0.252760% | 1.620296% |
| 49 | 0.203502% | 2.580187% | 0.645958% | 0.664931% |

Each error is modality-specific relative L2 over all real rows, never the
padding document. All values are finite. This confirms remaining propagated
differences despite the earlier local attention improvement. There is **no
same-geometry 256×256 default full-stack control yet**; the older 128×128
3.467995% result cannot establish a whole-stack benefit for this experiment.
The original series checks blocks only, not the newly captured heads/first
update. No trajectory/media-quality pass or production selection follows.

### Larger-geometry final-head/flow controls — 2026-10-12

The final-head reference initially failed before any numerical comparison:
its offline adapter assumed 1,024 packed rows even though this 256×256 capture
has 2,816 rows (2,785 real rows and a separate padding document). The adapter
now derives the packed extent from literal `documents.i32`, including the input
producer and unique-time ownership traversal. Compile-time shape regressions
cover both the earlier 1,009/1,024 and actual 2,785/2,816 geometries. Original
operator math and production execution are unchanged.

Failed root `/tmp/lunaflux-minimax-original-head-online64-256-20261012-v1`
is preserved. Corrected original replay and reports at
`/tmp/lunaflux-minimax-original-head-online64-256-20261012-v2` complete, exit
zero and release CUDA. Installed-module warnings remain in stderr.
Cluster-to-cluster input transfer keeps the complete native block-49 capture
unchanged and avoids downloading the entire stack to the desktop.

| Original operator on identical native input | Exact / elements | Relative L2 |
| --- | --- | --- |
| Final video head | 6,571 / 227,328 F32 | 0.000122690% |
| Final audio head | 383 / 13,248 F32 | 0.000135019% |
| First video RF update on native velocity | 227,328 / 227,328 F32 | 0%, bitwise |
| First audio RF update on native velocity | 13,248 / 13,248 F32 | 0%, bitwise |
| Initial video projection | 12,728,596 / 12,730,368 BF16 | 0.002212433% |
| Initial audio projection | 2,225,415 / 2,225,664 BF16 | 0.001398786% |
| Time embedding | 3 / 2,688 F32 | 0.001508693% |

All arrays are finite. These exclude a large local head/first-update error on
this capture, but do not compare native versus original propagated velocities,
prove complete conditioning parity or validate a full generated media trajectory.
The 2.580187% propagated block-49 video difference above remains unresolved.

A physical 1,024-row regression also completes at
`/tmp/lunaflux-minimax-original-head-geometry1024-regression-20261012-v1`.
Both modality head outputs and both first RF updates remain **bitwise identical**
to the earlier `original-head-actual-20261012-v3` outputs. The dynamic-geometry
repair therefore preserves the earlier smoke's observed arithmetic rather
than replacing its reference.

### Full 256×256 literal-noise trajectory dispatched — 2026-10-12

The larger native media control is now running, rather than stopping after one
evaluation. Exact `.178` user unit:
`lunaflux-minimax-joint-online64-original-noise-quality256-20261012-v1.service`.
Root: `/tmp/lunaflux-minimax-joint-online64-original-noise-quality256-20261012-v1`.
It reuses `f50b4ce3`'s committed executable, the same online64 AOT candidate,
actual original encoder caption and literal original video/audio initialization
as the completed full-prefix control. Configuration is 256×256, 120 requested
frames and 50 inclusive-zero sigma points / 49 denoising updates, with terminal
latent capture. No default production source or precision selection changes.

Startup preflight remains 82,890,073,512 planned bytes against the 96-GiB
device limit; the user unit has a 96-GiB memory cap, no swap and a 48-hour cap.
Its initial application stderr is empty. The earlier first evaluation required
43 minutes, so this is a long-running experiment; there is no measured complete
trajectory duration or quality result yet. Never overlap GPU work on `.178`.
`.179` remains available for serialized numerical comparisons.

After terminal retirement, collect actual output geometry and final latents,
render and inspect real video, assess audio separately, then compare installed
original VAEs on identical final latents serially on `.179`. Expected decoded
extent is 124 frames at 256×256, but report literal observed geometry and update
counts. Finite tensors and the earlier local head/update match are not a media
quality or complete numerical parity pass. The existing heartbeat remains
active with this exact unit and the unresolved correctness work recorded.

## User listening assessment — 2026-10-12

In response to the pending listening question about the earlier **native
MiniMax clip** shown in this conversation, the user reports: “natural coherent
sound, although i do not understand.” This is direct human listening evidence
for that previously displayed clip, not an automated assessment. It does not
establish intelligible speech, recognizable cat-related content, caption
alignment, or video quality. The still-running online64/original-noise 256×256
experiment has no terminal clip yet, so this feedback is **not** assigned to
that run. Earlier statements that listening was unassessed describe the state
before this feedback; audio and video conclusions remain separate.
