# Joint diffusion VAE decode reference boundary

This family-neutral package resolves and bounds VAE geometry proved by
`JointDiffusionPlan`. It also owns the request-bound F32 video-latent inverse
normalization contracts: video uses contiguous batch/channel/frame/height/
width storage and audio uses contiguous decoder-batch/channel/time storage.
Both apply the ordered F32 channelwise operation `latent * std + mean`.

The audio F32 dtype handoff, pointwise decoder-input projection,
weight-normalized `decoder.conv_pre`, first transposed-convolution upsampler,
and first alias-free AMP activation are modeled separately. The activation
keeps the upsample and downsample checkpoint filters distinct and preserves
the exact `1035 -> 2070 -> 1035` crop/padding geometry. The following
weight-normalized dilated convolutions, residual additions, parallel AMP-block
averaging, and later BigVGAN stages remain typed gaps, along with video
post-quant convolution, spatial tiling, temporal
chunking, transformer decode, overlap blending, and pixel postprocessing remain
ordered typed gaps. Numeric whole-decode entry points therefore still fail with
`DecodeSemanticUnavailable`; the scalar prefix is not a production fallback.
