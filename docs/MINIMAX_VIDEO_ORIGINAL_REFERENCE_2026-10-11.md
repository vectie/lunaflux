# Original MiniMax VideoVAE comparison — 2026-10-11

Later source comparison found the five-step joint runs below used the wrong
sigma grid: original H3 counts five F32 points including zero, hence four model
evaluations. The correction and original CPU comparison are documented in
`INDEPENDENT_CHECKPOINT_CORRECTNESS_2026-10-11.md`. This does not change the
zero-latent decoder comparison, but rules out labeling those earlier joint
runs original-schedule media-quality validation.

The source-selected FP16 operator boundaries in `b12c330f`, built with import
fix `79bf2a27`, were tested on real Spark hardware against the original
checkpoint's VideoVAE modules. This is a bounded decoder correctness experiment,
not a complete audiovisual quality pass.

The test-only C++/CPython bridge calls original `MiniMaxH3VideoVAE.from_pretrained`,
`decode_base`, and `processor.revert_tensor`. It does not reimplement decoder
math. Original F32 weights, latent mean/std and FP16 autocast are retained.
The original processor already returns NCTHW; no extra axis transpose is added.

Both executions consume standardized zero latent `[1,24,37,2,2]`. Original
temporal overlap produces **124 actual frames**, not the nominal 120, and both
canonical outputs have shape `[1,3,124,32,32]`: 380,928 finite F32 pixels.

| Native decoder | RMSE against original | Relative L2 | Maximum absolute error |
| --- | --- | --- | --- |
| Previous BF16 operator boundaries | 0.000608390 | 0.149646% | 0.004423857 |
| Source-selected FP16 boundaries | **0.000119524** | **0.029399%** | **0.000943661** |

Error falls approximately **5.1×**. The outputs are not bitwise equivalent.
The original reference runs on `.179` under a 32-GiB/no-swap limit and exits
zero without OOM. Native runs on `.178`; GPU ownership is released after both.
The original's diffusers configuration warnings are preserved, not called an
empty-stderr result. Generated native boundary probes independently pass
correctness, memcheck/leak, racecheck and synccheck on `.179`.

Native pixels: `/tmp/lunaflux-minimax-video-decode-fp16-20261011-v2/pixels.f32`.
Original pixels and both comparisons:
`/tmp/lunaflux-minimax-upstream-video-20261011-v3`.
Reproduction uses `scripts/run-minimax-upstream-video.mbtx` and
`scripts/compare-f32-tensors.mbtx`; the bridge is
`benchmarks/checkpoint_aten_reference/aten/video_upstream.cpp`.

Zero-latent agreement cannot establish caption fidelity, video coherence,
audio intelligibility, or the effect of remaining text-encoder differences.
Those require actual denoising inputs and independently assessed generated
media. The earlier 32×32/five-step colored output is not a quality acceptance
result. No acceptance threshold, runtime hashing, TLS or deployment hardening
was added.

## Generated-latent differential replay

`run-joint-latents` is an opt-in checkpoint diagnostic, not a serving path.
After successful terminal GPU completion it reads the final packed video/audio
states and applies the storage planner's existing inverse permutations on CPU.
It writes canonical **standardized** F32 latents beside the final media, before
mean/std conversion. The boundary arena is not assumed populated. Default
`run-joint` performs no extra readback; cancelled requests publish no latents.
Permutation regressions preserve all F32 bits for video patch and stereo layouts.

The original VideoVAE bridge can now consume that exact canonical video latent
instead of zeros. It still calls original decode/processor math under FP16
autocast; its CPU-only input probe verifies byte preservation and ownership.
The probe passes on Spark `.179` at
`/tmp/lunaflux-minimax-upstream-video-input-20261011-v1` (zero fixture, not a
generated-latent result). No full encoder/denoiser/decoder equivalence is implied.

Use the final `--capture-final-latents` option with
`scripts/run-minimax-joint.mbtx`, then prepare a **new** original reference root
with `scripts/run-minimax-upstream-video.mbtx prepare ROOT VIDEO_LATENT_F32`.
Capturing the actual denoiser output is necessary to distinguish encoder/denoiser
errors from decoder errors and to inspect media honestly. A 32×32/five-step
smoke run remains inadequate for semantic quality acceptance.

## Completed FP16 joint decoder ablation

Spark `.178` completed all fifty denoiser blocks over five flow steps and both
VAEs at `/tmp/lunaflux-minimax-joint-fp16-media-20261011-v1`: 1,523,712 video bytes
and 1,324,800 audio bytes, all finite. Unit lifetime was 1,807 seconds including
loading/preparation; `MemorySwapPeak=0`. The declared device-residency plan was
81,920,540,616 bytes. Cgroup peak was 36,347,752,448 bytes, which is **not** a
complete measurement of CUDA unified/device allocations.

This uses the same caption conditioning, five-step geometry and signed seed as
the earlier smoke run, changing only the VideoVAE AOT module's FP16 boundaries.
The remaining AOT modules are reused; this is an isolated decoder ablation,
not a claim that a new whole serving bundle was independently qualified.
Audio is bitwise unchanged at all **331,200 F32 values**. Video differs with
RMSE 0.000867734, relative L2 0.243979%, maximum absolute change 0.008955061.
Those are native-to-native changes, **not** error against the original decoder.

The downloaded 124-frame, 32×32 contact sheet remains blurred colored regions;
it does not visibly establish the requested "A red cat". Finite pixels,
unchanged audio, and the earlier zero-latent decoder improvement do not constitute
media-quality acceptance. Latent capture and original replay on generated inputs
are the next isolation step; meaningful resolution/step coverage remains open.
No audio intelligibility assessment is claimed from waveform statistics alone.
Preview: `/tmp/lunaflux-minimax-fp16-media-preview-20261011-v1/frames.png`.

## Prompt presentation cross-check

The installed original H3 pipeline calls `minimax_h3_text_only_ids` for `t2va`.
That function explicitly tokenizes the prompt verbatim without special tokens;
the pipeline supplies one text tag per token. The existing three-token
"A red cat" conditioning is therefore not missing a chat template. Adding
chat-format tokens would change the original model input, not fix it. This is
a source-level presentation check, not proof that the encoder's remaining
numerical differences are harmless or that tiny smoke-run media is acceptable.

## Generated-latent replay support

The selected compensated-square/ordered-dot caption now feeds a new joint replay
at `/tmp/lunaflux-minimax-joint-latents-selected-20261011-v1`, with the existing
five-step seed and FP16 VideoVAE module. Commit `98d5e90b` enables terminal-only
canonical latent capture; the other joint AOT modules are unchanged. Its declared
residency budget remains below 96 GiB, and no GPU workloads overlap on `.178`.
That replay completed and released its GPU; its five old-schedule evaluations
make it a decoder-isolation fixture, not an original-schedule quality result.

`scripts/collect-minimax-joint-output.mbtx` can explicitly collect the canonical
latents with `--latents VIDEO_LATENT_BYTES AUDIO_LATENT_BYTES`, preserving the
separate layout metadata. For this bounded geometry the byte counts are 14,208
for standardized video `[1,24,37,2,2]` and 52,992 for standardized audio
`[2,32,207]`. This is a diagnostic geometry, not a restriction on production
request shapes.

`scripts/run-minimax-upstream-audio.mbtx` now complements the original VideoVAE
replay. Its test-only C++ bridge calls the installed original H3 AudioVAE adapter,
which retains FP32 weights/decoding, original mean/std conversion, and canonical
stereo waveform layout. It does not reimplement audio arithmetic. The two bridges
share a CPU-owned, byte-preserving canonical F32 input loader. CPU extent/ownership
tests and the audio AddressSanitizer probe pass; leak detection for third-party
Torch/CPython process lifetime is disabled, so this is not an LSAN result.
The native suite remains **5,003/5,003**, with the documented preexisting
toolchain warning exclusions. Generated-latent GPU comparisons and meaningful
media quality remain open until the corresponding actual outputs are inspected.

The original AudioVAE has now decoded this captured standardized latent. All
331,200 waveform values are finite; native versus original has **0.071764%**
relative L2 and maximum absolute error 0.000357319. Evidence is under
`/tmp/lunaflux-minimax-upstream-audio-generated-20261011-v1`. A separate matched
destandardized-zero replay measures 0.053442% relative L2. The earlier 97.5%
zero comparison used different standardized/destandardized inputs and is not
decoder-error attribution. Generated VideoVAE replay is separate; neither
decoder comparison by itself validates caption fidelity or audio quality.

Corrected-schedule joint replay uses committed `de9e1ec8` under
`/tmp/lunaflux-minimax-joint-sigma-20261011-v1`. Fresh AOT preparation is at
`/tmp/lunaflux-minimax-request-aot-sigma-20261011-v3`: 27 unchanged module sources
reuse their prior CUBINs; four changed module sources are recompiled. The request
contains five original F32 grid points and therefore **four** model evaluations.
Its completion and media inspection remain required; dispatch is not a pass.
