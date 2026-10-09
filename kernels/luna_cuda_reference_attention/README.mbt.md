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

The wrappers specialize head dimension 128, GQA ratios 1/2/4/8, the raw-CSR
ten-pointer metadata ABI, and canonical contiguous page
stride. Unsupported shapes are explicit errors, not silent compiler-kernel
fallbacks. The bundle exporter has an explicit
`--prefill-reference-flashattention` AOT opt-in. This is not a global default
or a performance promise.

`ReferencePrefillSchedule` retains two Q64 terminal alternatives: KV128
(80 KiB shared) and KV64 (48 KiB shared). The same immutable schedule owns
source specialization, rounded key extent and launch resources; a C++ static
assert checks the pinned dependency's storage trait. The candidate exporter
emits both without selecting one. Defaults remain KV128. Select only after
paired whole-service measurement for the target device/workload; a smaller
shared footprint is not itself a performance result. KV64 is rejected for the
independently defined decode backends. Both prefill slots must bind identical
launch resources and the same module, preserving the single-writer contract.

These alternatives retain the same BF16-probability numerical permission, but
different reduction tiles need not produce identical bits. They require
independent accuracy and sanitizer checks, not a bitwise-equivalence claim.

FlashAttention's bounded CSR copy requires page size divisible by eight; this
constraint is not imposed on the independent FlashInfer decode layout.

`FlashAttentionMixed` owns every active output row when at least one row is
prefill; it writes nothing in a decode-only invocation. The admitted module
supplies a backend-neutral `AttentionMixedRowCoverage` value to executor
planning. A mixed graph therefore either uses one all-row writer, or a
prefill-only writer plus a disjoint decode companion, never both. Pure decode
continues to use its independently prepared implementation. The separately
named prefill-only wrapper remains available for that narrower domain.

The numerical permission explicitly names BF16 probabilities and base-two
exponentials; it cannot be admitted as the old F32 exponential-only rewrite.
Offline decode measurements must bind the new bundle scope. Reference mode
admits only baseline/ordinary/split-decode route alternatives, not the old
partitioned-prefill companions. No tuning, filesystem checks, allocations or
backend discovery are added to the token step.

The current BF16 wrappers have passed independent scalar correctness,
determinism, read-only/inactive-output checks and GPU memcheck/leak,
racecheck and synccheck on GB10. These kernel tests do not establish serving
integration or support for a different dtype. See
[precision and experiment scope](../../docs/REFERENCE_ATTENTION_PRECISION_2026-10-04.md).

`benchmarks/gpu_pipeline/prepare_reference_attention.mbtx` prepares a new
inference-only copy of the pinned upstream headers. It removes unused host
framework RNG dependencies and omits nullable LSE stores. Its bounded tail
resolver also prevents speculative final-segment addressing from reading past
an exact-size CSR page table; padded page tables are not assumed. These changes
do not change the core attention arithmetic. Original dependency sources
remain untouched. Header hashes and the adaptation marker are checked before
compilation.

See [the serving integration experiment](../../docs/REFERENCE_ATTENTION_INTEGRATION_2026-10-08.md)
for the initial regression, the selected-trace diagnosis and the retest.
