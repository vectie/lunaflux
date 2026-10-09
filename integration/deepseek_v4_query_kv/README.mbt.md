# Checkpoint-backed quantized query/KV prefix

This adapter supplies official DeepSeek tensor names and model dimensions to
the model-neutral `QuantizedQueryKvPrecision` plan. CUDA details remain in
`kernels/quantized_query_kv_cuda_source`; prepared effects and deterministic
release remain in `integration/quantized_query_kv_frame`.

The six stages are Query-A, learned query RMSNorm, Query-B, unweighted headwise
normalization, shared-KV projection, and learned KV RMSNorm. Projection matrices
retain E4M3 payload and block-128 UE8M0 scales; norm vectors retain BF16. The
unweighted head norm preserves the reference's BF16 square/mean/epsilon/inverse/
product boundaries, unlike learned RMSNorm's F32 arithmetic and final BF16 round.

Preparation accounts for all five compact checkpoint banks plus six scratch
buffers before upload. The caller borrows the six launches into its decoder
queue; there is no extra per-layer submission or completion. Release the queue
first, then this owner, then its external operands/context. Partial preparation
can be closed deterministically. No file reads, authentication, source generation
or scratch allocation occur in the token-step path.

This is a single-rank, unsharded projection prefix, **not a complete attention
layer or DeepSeek/DSpark model runner**. Rotary, KV quantization simulation,
sliding-window/compressed caches, index selection, attention, output projections
and distributed head ownership are subsequent integration work. No performance
claim follows from the small numerical fixture.
