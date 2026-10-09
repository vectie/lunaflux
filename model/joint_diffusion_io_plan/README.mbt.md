# Joint diffusion I/O plan

This family-neutral package is LunaFlux's immutable boundary between validated
joint audio/video generation inputs and later execution. It binds exact model
content and plan identity, normalized conditioning order, resolved media and
latent shapes, seed, inference steps, unit distilled guidance, cancellation
policy, and terminal result metadata into a deterministic request digest.

Inputs are descriptors for content that has already been decoded, validated,
tokenized, and normalized to the requested canvas/rate. This package does not
read files, decode codecs, preprocess media, schedule work, materialize tensors,
run a transformer or VAE, select a device, or expose a transport protocol.
