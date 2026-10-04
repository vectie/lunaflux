# Precision support and the native reference attention experiment

BF16 is an existing execution format, not a proposed feature. Current matched
Qwen serving comparisons use BF16 weights, activations, KV and output. Adding
a reference attention backend must preserve that support. It must not make
the model, scheduler or compiler semantic IR depend on a BF16-only library.

## Independent numerical roles

The immutable numeric plan must distinguish:

1. Weight storage: format, packing, layout and scale/zero-point metadata.
2. Graph activations and an operation's internal compute operands.
3. Accumulator and probability representation and arithmetic policy.
4. KV storage: key/value dtype, page layout and persistent scale ownership.
5. Output conversion and rounding.

For example, FP8 weights with BF16 activation and KV boundaries are not an
FP8-cache implementation. FP32 accumulation is not FP32 weight storage.
The reference FlashInfer wrapper's `f32` numerical-law suffix describes its
probabilities; its Q/K/V/output pointers are BF16.

`model/numeric_contract` remains the owner of backend-neutral tensor and
operation representations. `kernels/luna_cuda_reference_attention` now uses
that vocabulary explicitly for query, KV, output and accumulator roles.
Its first terminal specialization accepts BF16/BF16/BF16/F32 and rejects
other combinations at startup. It neither invents a second generic dtype
enumeration nor silently converts cache pointers.

## Support must be stated per path, not per format name

| Format | Current scope | Work that cannot be inferred from that scope |
| --- | --- | --- |
| BF16 | Existing Qwen serving; new native reference-attention correctness checks | The new reference modules are not yet admitted into the serving bundle |
| Finite FP8 E4M3 | Numeric contracts, loaders and bounded CUDA implementations/qualification paths | General Qwen FP8 serving and FP8 KV are not established by BF16 attention tests |
| Block FP8 / UE8M0 | Additional software contracts and source work exist in the working tree | Not a completed, benchmarked production path; do not conflate scale/layout variants |
| INT8 | Weight-only numerical contracts and bounded execution paths | Not equivalent to every W8A8, affine or KV quantization scheme |
| FP16, FP8 E5M2, FP4/NVFP4/MXFP4, packed INT4 | Further implementation work required | Upstream-library support or parser recognition alone is not engine support |

The modern-format implementation direction is BF16 baseline retention, then
complete FP8 paths, then explicitly packed FP4/INT4 paths. FP8 encodings must
not alias each other. FP4 variants require their exact block/group scale,
packing, rounding and saturation semantics, not a generic “four-bit” flag.
Weight-only and activation/cache quantization should remain independently
selectable. Device-specific instructions and capability checks belong only
in backend lowering/selection.

## Functional compiler boundary

Keep the path as pure semantic numeric contracts → pure storage/compute and
layout plans → explicit memory/effect plan → device lowering → immutable AOT
artifact. Precision/layout/device identities participate in the offline
selection key. Conversions and persistent KV scale writes are explicit
effects, not hidden pointer casts or request-time library planning.

The current reference experiment translates the engine's existing CSR page
metadata to native device-library views without gathering KV. Original
headers are retained; an inference-only FlashAttention copy removes unused
ATen RNG dependencies and makes unused LSE stores nullable. The core attention
arithmetic remains upstream. FlashAttention decode groups adjacent GQA query
heads as query rows to reuse a KV tile; this is CUDA lowering, not a model
special case.

The pinned FlashInfer decode header SHA-256 is
`019d673aa848a938798a2c58b34b9b5813a3f137962cbbd90ef7bd71f636f373`,
identical to the retained reference counter capture's header. FlashAttention
uses the previously captured pinned vLLM image source. No production Python,
PyTorch, runtime JIT or second scheduler is introduced.

## Validation scope

On the NVIDIA GB10, native BF16 wrappers compile for sm121 and are tested
against an independent scalar oracle, deterministic replay, inactive-output
preservation and read-only Q/K/V checks. GPU work is serialized inside an
8-GiB/no-swap scope with a 32-GiB host-memory reserve. Memcheck includes leak
checking; racecheck and synccheck cover the reference device pipelines.

The earlier prototype's head-by-head FlashAttention decode is retained as
an experiment, not promoted. The grouped-head revision passed all seven
numerical cases and memcheck/leak, racecheck and synccheck for each of the
three modules. Paired measurements use the actual compiler module symbol
from the existing serving artifact, not the similarly named unbound
candidate symbol. Five local reference-package tests pass; the complete
native suite passes 4,367/4,367 with the GQA regression included.

At 16 rows, one query per row and history 4096 with the same fragmented pages,
five alternating warm event-time trials gave these medians:

| Pair | Existing compiler µs | Reference µs | Reduction |
| --- | ---: | ---: | ---: |
| Compiler / grouped FlashAttention | 1195.232 | 1184.098 | 0.9% |
| Compiler / FlashInfer | 1188.646 | 1172.176 | 1.4% |

Every pair uses identical BF16 Q/K/V and independent scalar checking.
Maximum pairwise absolute differences were 0.000488281 and 0.000244141;
the numerical law is not claimed bitwise-identical. These are unpartitioned
decode kernel probes, not reference servers, prefill-chain results or output
token-rate measurements. They do not establish that simply replacing
attention with a library will eliminate the serving gap.

This is native kernel qualification, not a new end-to-end serving comparison
or an FP8/FP4 performance claim. Remaining work is the explicit startup
module/bundle contract and serving selection, followed by an attention-only
LunaFlux A/B campaign. Do not label the adapter fully integrated until that
selected-path comparison executes.
