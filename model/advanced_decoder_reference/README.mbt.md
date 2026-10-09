# Advanced decoder correctness reference

This package is an allocation-using, bounded Float oracle for semantic
primitives in `AdvancedDecoderExecutionPlan`. It exists for deterministic
fixtures and differential qualification, never as a production fallback.

It covers expert score transforms, stable biased top-k selection, token-hash
routing, routed/shared expert combination, and the versioned mHC block-control,
pre-reduction, post-residual combination, and head-reduction phases. The mHC
oracle binds BF16 activation boundaries widened to `Float`, F32 control weights
and intermediates, per-row softmax and alternating normalization, and the final
BF16 rounding boundary without pretending to perform that physical conversion.
It also covers learned and deterministic compressed-index selection,
MTP/DSpark stage ordering, and a tiny end-to-end block composing RMS
normalization, learned sparse attention, hyper-connection phases, stable MoE
routing, and selected SwiGLU experts. It
does not implement full-profile low-rank attention, quantized arithmetic,
weight materialization, KV state, scheduling, device execution, or kernels.
