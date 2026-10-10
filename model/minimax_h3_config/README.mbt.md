# MiniMax H3 configuration admission

This family-owned package parses bounded official MiniMax H3 configuration
documents without routing through the decoder-oriented shared config reader.
It rejects duplicate keys, excessive UTF-8 size or nesting, unknown fields,
non-finite numerics, and arbitrary executable metadata. Original AudioVAE and
VideoVAE entry points recognize their known descriptive `auto_map` value without
importing or executing it; their source data determine the numeric contract.

Schema v1 admits exact original and Diffusers transformer layouts, original
FL2VA/Ref2VA pipeline indexes, converted video/audio VAE geometry, and the
released video/audio scheduler shifts. Video VAE parsing retains the exact 24
F32 latent means and standard deviations used by inverse normalization rather
than merely validating and discarding them. Parsing authenticates document
semantics only; artifact digest verification remains the caller's boundary.
Original VideoVAE wrapper/source JSON and AudioVAE wrapper/metadata/restricted
source YAML resolve to these same numeric records. No backend or scheduler
package parses model-family configuration.
