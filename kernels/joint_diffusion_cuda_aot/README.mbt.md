# Joint diffusion CUDA AOT candidates

This package lowers exact request-shape-specialized video/audio latent-input
projections, selected-row final RMSNorm, final output heads, and shifted Rectified Flow Euler updates into
deterministic CUDA source. It binds the joint-diffusion plan identity, requirement digest,
request geometry, device target, compiler policy, launch shape, and
artifact-compatible operand contract into a canonical recipe.

Projection inputs, row-major `[output,input]` weights, and separate bias
buffers are F32; projected hidden rows are BF16. Each output uses ordered F32
multiplication and addition before one round-to-nearest-even BF16 conversion.
Video rows are batch/frame/patch-row/patch-column major, with vector lanes in
channel/temporal/height/width order. Audio rows are channel-major then time.

Final RMSNorm consumes BF16 selected hidden rows and the BF16
`norm_out.norm.weight`, reduces each row in ordered F32 with the plan's exact
rational epsilon, and rounds once to BF16. This is deliberately only the
RMSNorm sub-operation. A following candidate consumes already row-aligned BF16
shift and scale, stages `hidden * (1 + scale) + shift` with BF16 rounding, and
widens the final BF16 value to the F32 output-head input. Projection of the F32
timestep embedding through BF16 `norm_out.linear` and row-to-timestep selection
of shift/scale remain separate missing operations.

A repeatable one-row parameter candidate now covers the exact
`norm_out.linear` prefix from an already produced F32 timestep embedding to one
BF16 shift/scale table row. Runtime distinct-row orchestration and packed
`timestep_indices` selection are deliberately outside that candidate.

Output heads consume final-normalized modality-selected F32 hidden rows,
row-major F32 `[output,input]` weights, and F32 bias, and produce F32 velocity
predictions with ordered F32 multiplication and addition. Selecting retained
rows before this row-independent dense projection is exactly equivalent to the
official project-then-select order.

Shifted-flow `FlowCoefficientsInput` is exactly three prepared F32 cells:
`[1 - timestep, next_sigma / sigma, 1 - next_sigma / sigma]`.
State, velocity, and output remain F32. The generic rectified-flow renderer
computes `denoised = state + sigma_t * velocity`, then
`ratio * state + complement * denoised`, using explicit rounded operations.
There is no BF16 round-trip, per-element division, or tensor-sized scratch.
The v2 recipe and new operand role distinguish this ABI from the obsolete
two-sigma BF16 update. FMA contraction and reassociation are rejected.

Candidates remain offline-only and non-bindable. Compilation, catalog
admission, driver loading, execution, numerical qualification, and performance
promotion remain separate authorities.

The video-VAE prefix candidate consumes F32 contiguous
batch/channel/frame/height/width latents plus ordered F32 standard-deviation and
mean tables. It performs only ordered inverse normalization. Decoder dtype
casting, `post_quant_conv`, chunk/tile orchestration, transformer decode,
blending, and pixel postprocessing remain outside its claim.

The audio-VAE prefix reuses the same family-neutral affine renderer for exact
contiguous `[decoder_batch,channel,time]` storage. Its ABI is normalized F32
latents, 32 ordered F32 standard deviations, 32 ordered F32 means, and F32
destandardized output. It ends before the official decoder dtype cast and
`dec_in_proj`.

The following AudioVAE candidate covers only the official F32
`dec_in_proj.weight [2048,32,1]` and bias `[2048]`. It preserves contiguous
`[2,32,207] -> [2,2048,207]` channel-major ordering and stops before BigVGAN's
weight-normalized `decoder.conv_pre`.

Subsequent exact candidates cover `decoder.conv_pre`, the first
weight-normalized transposed-convolution upsampler, and the first complete
alias-free activation. The activation uses three ordered F32 launches and two
separate 2,119,680-element workspaces for `[2,512,1035] -> [2,512,2070] ->
[2,512,1035]`. The dilated convolutions, residual path, parallel AMP-block
averaging, and remaining decoder stay outside the claim.
