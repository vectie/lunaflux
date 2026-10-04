# Native reference attention lowering

This CUDA backend package emits immutable AOT wrappers around pinned
FlashAttention prefill/decode and FlashInfer decode device implementations.
It does not own model selection, request scheduling, KV allocation, Python,
runtime JIT, or a host-framework execution plan.

The input ABI is read-only paged attention after a separate committed KV-write
effect. Queries are contiguous causal suffix chunks; keys and values use the
engine's existing separate BF16 page pools. No cache gather is required.

## Precision boundary

Query, KV, output, accumulator and internal probability precision are distinct.
The initial executable specialization uses BF16 Q/K/V/output and F32
accumulation. FlashAttention rounds the probability operand to BF16;
FlashInfer decode retains F32 probability. Their numerical law identities
are deliberately different. `f32` in the FlashInfer numerical law refers to
probabilities, not to its query, cache or output storage.

Projection weight format does not belong to the attention ABI. FP8 or INT8
weights may feed BF16 attention after their independently defined projection
and scale conversion. They do not imply FP8 or INT8 KV support. The constructor
uses the existing backend-neutral numeric vocabulary for all operand roles,
and rejects unsupported combinations before source generation. A future FP8
cache specialization must own its scale pointers, layout and conversion
semantics; a BF16 pointer cast is not such a specialization.

FP4/NVFP4/MXFP4 and packed INT4 additionally require explicit packing and scale
contracts. This package does not claim those production implementations, or
equate different FP8/FP4 encodings. The architecture is not BF16-only; this
first reference experiment is BF16 to match the existing serving comparison.

## Current scope

The wrappers specialize head dimension 128, GQA ratios 1/2/4/8, the legacy
ten-pointer metadata ABI, and canonical contiguous page stride. Unsupported
shapes are explicit errors, not silent compiler-kernel fallbacks. New source
rendering is not yet a serving admission or a performance result.

The current BF16 wrappers have passed independent scalar correctness,
determinism, read-only/inactive-output checks and GPU memcheck/leak,
racecheck and synccheck on GB10. These kernel tests do not establish serving
integration or support for a different dtype. See
[precision and experiment scope](../../docs/REFERENCE_ATTENTION_PRECISION_2026-10-04.md).

`benchmarks/gpu_pipeline/prepare_reference_attention.mbtx` prepares a new
inference-only copy of the pinned upstream headers. It removes unused host
framework RNG dependencies and omits nullable LSE stores; it does not change
the core attention arithmetic. Original dependency sources remain untouched.
