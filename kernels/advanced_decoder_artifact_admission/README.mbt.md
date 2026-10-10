# Advanced decoder artifact admission

This family-neutral package joins an exact
`AdvancedDecoderKernelRequirements` set to content-addressed module bytes,
stable catalog AOT entry-point identities, bounded function symbols, an exact
device target, launch dimensions, and an ordered advanced-decoder operand ABI.
It imports no model family, scheduler, KV, API, device runtime, or native CUDA
package.

Module digests are supplied labels, not checksums authenticated by admission.
No CUBIN scan occurs here; bounded sizes, unique labels, required symbols and
semantic launch compatibility remain checked. Optional payload integrity
verification belongs outside engine startup and inference.

The existing catalog capability and launch operand vocabularies describe the
current dense decoder graph and cannot faithfully name MoE, compressed
attention, hyper-connection, token-hash, MTP, or DSpark operations. This
package therefore reuses only faithful identities and evidence types:
`DeviceTarget`, `AotArtifactDigest`, `AotKernelEntryPoint`,
`KernelEntryPointInput`, symbol validation, and `AotLaunchDimensions`. It does
not alias advanced capabilities into the current generic kernel vocabulary.
The expert-score ABI is exactly counts, F32 routing-score input, and F32
routing-score output; admission cross-checks both score byte ranges against the
declared routed-expert width and launch row count. No model-weight operand is
invented for this elementwise transform.
The paired Q/K RMSNorm ABI is exactly counts, query input, query norm weights,
query output, key-value input, key-value norm weights, and key-value output.
Admission checks the two independent BF16 widths, the bounded row capacity,
and the exact `[rows,2,1]` launch geometry.
The token-hash ABI is counts, I32 token IDs, the row-major I32 lookup table,
and row-major I32 expert-index output. Admission checks the complete
vocabulary/top-k table extent, row capacity, 4-byte alignment, and the exact
one-block-per-row launch geometry.
Stable top-k selection has separate counts, original F32 scores, F32 selection
bias, and I32 selected-index output operands. Selected-weight finalization has
counts, original scores, selected indices, and F32 selected-weight output, so
hash lookup cannot be mistaken for score-based selection.
Deterministic compressed indexing has a distinct five-operand ABI: counts,
I32 query positions, I32 cache offsets, I32 selected counts, and a dense I32
slot matrix. Admission binds the ratio, maximum-position-derived row width,
4-byte alignment, exact bytes, and one-block-per-row launch shape; it does not
reinterpret hidden-state or routing-index roles as cache topology.
The deterministic window/compressed join uses eight exact I32 roles: counts,
query positions, sequence lengths, prefill/decode modes, compressed counts,
compressed rows, joined counts, and joined rows. Admission binds maximum
compressed and joined widths to position, window, and ratio geometry.
Non-overlapping compressed-K/V preparation has seven exact roles: counts,
prefill/decode modes, sequence lengths, borrowed ordinary BF16 K/V, borrowed
compressed BF16 K/V, prepared lengths, and prepared BF16 K/V. Admission binds
both window and ratio plus maximum-position/head geometry and never treats the
buffers as owned cache state.
Selected sparse attention has eight directional roles for counts, query,
query-batch ids, shared K/V, cache lengths, sink, joined indices, and output.
Admission derives selected and K/V extents from window, compression ratio, and
maximum position geometry; it does not absorb the preceding join/preparation.
Output inverse RoPE separately binds counts, positions, and one in-place
full-head BF16 buffer, with exact head/position geometry and no model weights,
workspace, or out-of-place alias contract.

Query-A admits only its exact five-operand replicated block-E4M3/UE8M0 ABI:
counts, BF16 hidden input, row-major E4M3 codes, raw UE8M0 block scales, and
BF16 rank output. The separate replicated KeyValue projection admits the same
block-FP8 storage semantics with an exact five-operand ABI and fixed
`hidden→512` pre-normalization output. Query-B remains typed unavailable, so
complete admission advances through KeyValue and fails closed at ordinal 4;
generic dense roles cannot stand in for either exact candidate.
The independently versioned attention output-A operation admits counts plus
directional BF16 input, converted group-major weight, and rank-output operands
with exact bytes and alignment. Output-B remains explicitly unsupported.

The versioned mHC block-control requirement has an exact eight-operand ABI:
counts, BF16 stream state, separate F32 function/base/scale inputs, and separate
F32 pre/post/combination outputs. Admission authenticates its row capacity,
hidden/stream geometry, output order, and launch shape. Directional
pre-reduction and destination-major post-combination have distinct ABIs. Head
reduction binds BF16 state, F32 function/base/scalar-scale inputs, and one BF16
reduced output. Their exact ABIs remain canonicalized, while complete-model
admission still fails earlier at Query-B.

Admission authenticates and canonicalizes inert startup evidence only. Module
bytes, their caller-supplied digests, and optional declared qualification
digests remain `InertUnqualified`. The result cannot load a module or launch a
kernel, and `require_execution_authority` always fails until a separate opaque
qualification authority exists.
