# Original MiniMax VideoVAE comparison — 2026-10-11

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
