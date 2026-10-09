# MiniMax H3 CUDA AOT integration evidence

This test-only package joins official FL2VA and Ref2VA metadata to the
family-neutral joint-diffusion requirement and CUDA-candidate packages. Keeping
the join here prevents MiniMax types from entering reusable kernel or runtime
owners while proving the official profiles produce exact request-specialized
video/audio latent-input projection, final-RMSNorm, output-head, and
shifted-flow geometry.

The projection evidence covers the official mixed-precision boundary: F32
packed inputs, separate F32 weight and bias operands, and BF16 hidden output.
The output-head evidence covers the distinct official F32 boundary:
final-normalized selected F32 hidden rows, `proj_out`/`audio_proj_out`-shaped
F32 weights and bias, and F32 velocity predictions.

The final-RMSNorm evidence separately covers BF16 selected rows, the exact
BF16 `norm_out.norm.weight`, the official `1/100000` epsilon, and BF16 output.
The row-aligned AdaLN affine stage and BF16-to-F32 handoff are covered by a
separate exact candidate. It remains non-bindable because distinct-timestep
row orchestration and packed row-to-timestep shift/scale selection are still
missing execution operations.

One-row parameter-table evidence additionally binds F32 `[2688]`
conditioning, BF16 `[10752,2688]` weight and `[10752]` bias, and the two BF16
`[5376]` shift/scale halves for both official variants.

Both profiles also bind exact VAE inverse-normalization prefixes. Audio uses
the official contiguous `[2,32,207]` decoder input, 13,248 F32 elements, and
keeps the remaining decoder body uncovered and non-executable.

The AudioVAE decoder-input projection evidence then binds the official F32
`dec_in_proj` tensors and exact `[2,32,207] -> [2,2048,207]` pointwise geometry.
The remaining weight-normalized BigVGAN body is still a typed coverage gap.
