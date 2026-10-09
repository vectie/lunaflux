# Joint diffusion kernel requirements

This package derives an immutable, ordered, digest-bound semantic and AOT
requirement vocabulary from a family-neutral `JointDiffusionPlan`. It records
conditioning, latent-input and modality-output projections, final row-wise
RMSNorm, joint-transformer work, shifted-flow steps, exact F32 video/audio
latent-destandardization prefixes, and separate remaining decoder bodies with
specialization-relevant geometry. Final RMSNorm requirements authenticate
hidden width and the exact rational epsilon. The following boundary separately
authenticates row-aligned BF16 shift/scale modulation and F32 head input; it
does not claim timestep projection or packed row-to-timestep selection.

The requirements digest also authenticates the distinct input- and
output-projection storage ABIs and packed video/audio row layout. This prevents
a kernel candidate from reinterpreting channel-first latent memory as already
packed rows or weakening an F32 output head into BF16.

Video VAE preprocessing is deliberately split: an exact F32
batch/channel/frame/height/width inverse-normalization requirement precedes a
separate decoder-body requirement. Covering the prefix cannot satisfy the
post-quant convolution or the rest of the decoder.
Audio preprocessing likewise binds the official contiguous decoder-batch/
channel/time layout. Stereo is represented as decoder batch 2; this is not a
generic request-batch or output-channel equivalence. The following exact F32
pointwise decoder-input projection is separate from the remaining BigVGAN
decoder body. This vocabulary is domain-separated as requirements encoding v4.

The values are requirements only. The package imports no device backend,
kernel implementation, artifact catalog, scheduler, or request protocol. It
does not prove that an AOT module exists or that any operation is executable.
The current decoder kernel vocabulary is rejected for every diffusion
operation rather than being silently aliased.
