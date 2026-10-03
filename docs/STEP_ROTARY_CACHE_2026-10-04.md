# Immutable step rotary operands

The full attention-ingress alternative can prepare the rotary sine/cosine pairs
once per submitted token domain and borrow them across every shape-identical
layer. This is an explicit AOT alternative, not an implicit ABI change.

## Compiler and execution ownership

`compiler/elementwise_physical_ir/StepRotaryCachePlan` describes the bounded
token/pair domain and F32 storage size. CUDA lowering emits the preparation
entry point and cached consumer in the same translation unit and CUBIN.
Preparation retains the existing `sincosf(__fmul_rn(position, inverse_frequency))`
law; consumers load the exact resulting F32 pair. No approximate exponential or
rotary law is introduced by this change.

The runtime appends `maximum_tokens * (head_dimension / 2) * 8` bytes to the
already-owned rotated-QKV sidecar. It prepares arguments and resolves both
functions at startup. One preparation effect is prepended before all layer
consumers on eager, baseline, decode, wide-prefill and partitioned-prefill
ordered graphs. Captured buckets cap its grid using the live token bound.
The pair region may be overwritten only by the next ordered step, after the
previous graph's consumers retire. It adds no allocation, validation, filesystem
access, trigonometric host work or JIT to the token path.

## Explicit identities

- Candidate and runtime bundle exporters both accept `--step-rotary-cache`.
- Consumer symbol: `lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3`.
- Preparation symbol: the consumer symbol followed by `_rotary_prepare`.
- Consumer ABI: existing fifteen operands followed by `const float2* rotary_pairs`.
- Preparation ABI: `step_counts`, `query_positions`, writable `rotary_pairs`.
- Runtime ABI: `attention-ingress-prepared-rotary-production-v11`.
- Runtime bundle schema: `lunaflux-reusable-fused-runtime-bundle.v10`.

Older bundles retain their original symbols and fifteen-operand ABI. A cached
consumer cannot be selected by relabeling an older bundle schema, or without
the full-ingress route. Both kernel source and recipes have distinct digests.

## Verification boundary

Host tests cover bounds, unique startup preparation identity, prepend ordering,
live-token grid capping, schema downgrade rejection, CLI propagation and the
single rotary preparation site in generated source. A physical differential
test must verify cached versus uncached values, include one preparation per
whole layer chain in timing, and verify actual selected symbols before a
production selection change. Default CLI selection remains uncached pending
that qualification; source construction alone is not a speed or deployment
claim.
