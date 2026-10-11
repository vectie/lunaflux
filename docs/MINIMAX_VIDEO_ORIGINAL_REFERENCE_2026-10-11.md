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
