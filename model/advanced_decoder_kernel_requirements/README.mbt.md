# Advanced decoder kernel requirements

This package derives a bounded, immutable, first-use-ordered set of semantic
operations and required AOT capability names from an
`AdvancedDecoderExecutionPlan`. Its digest binds the exact content digest,
execution-plan digest, operations, geometry, and capability order.
The immutable result also exposes the complete model-wide geometry required by
offline lowerers; consumers never need to recover hidden width, vocabulary,
head shape, or position bounds from a family type.
The expert-score operation additionally carries its routed-expert width because
that value changes both AOT work geometry and the routing-score tensor ABI.
The paired query/key RMSNorm operation likewise carries the independent query
low-rank and key-value head widths. This keeps exact activation and weight
geometry in the authenticated requirement rather than recovering it from a
model-family type during lowering.
Token-hash routing carries the hash-layer count, vocabulary size, selected
expert count, and routed-expert count. These values authenticate the complete
I32 lookup-table and output geometry as well as the valid expert-index range.
Non-hash top-k selection separately authenticates routed width and selected
count. Selected-weight finalization carries routed width, selected count,
normalization policy, and scaling, and consumes indices produced by either
stable score selection or token-hash lookup without conflating their ABIs.
Query low-rank attention is ordered as query-A projection, the separately
owned Q/K RMSNorm requirement, then query-B projection. Query-A derives
`[rank,hidden]`; query-B authenticates rank, query-head count, and head
dimension for its `[heads*head_dimension,rank]` output projection.
Attention output projection is likewise split. Output-A authenticates rank,
groups, heads, and head dimension for group-local converted-BF16 work;
Output-B authenticates rank, groups, and hidden width and remains an
independent block-FP8 requirement.
Versioned mHC requirements keep block control generation, pre-reduction,
post-residual combination, and head reduction separate. They authenticate the
stream width, iteration count, control epsilon, and RMS normalization epsilon;
block control geometry derives `6 * width` without a model-family dependency.

The package performs capability planning only. It does not select, qualify,
load, or execute artifacts. Its projection to the existing dense generic
kernel vocabulary is deliberately lossless and therefore rejects both query
projection stages, low-rank
attention, scaled RoPE, versioned mHC phases, compressed attention, MoE, and auxiliary
prediction rather than substituting superficially similar dense operations.

There are no scheduler, KV-cache, API, device, native ABI, CUDA, or kernel
catalog dependencies.
