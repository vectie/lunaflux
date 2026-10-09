# DeepSeek V4 logical numeric planning

This package builds a bounded logical numeric plan for all five closed
DeepSeek-V4 profiles. It coalesces the exact official tensor manifest into
ordered role/storage runs whose counts expand to the profile's exact tensor
cardinality, up to the family-owned maximum of 145,118 tensors. The manifest,
source execution plan, operation schema, and final binding each have joined
content-addressed identities.

Plain BF16/F32/I64 roles, FP8 E4M3 block-128 parameters, UE8M0 block-scale
metadata, packed FP4 E2M1 block-32 expert parameters, and the canonical E2M1
codebook are distinct semantics. The generic `numeric_contract` is reused for
ordered operation execution only where its compute vocabulary is faithful.
Packed FP4 expert execution remains explicitly unsupported. Token-hash routing
records the required checked I64-checkpoint-to-I32-compute conversion before an
exact row-major lookup; no current materializer grants that conversion or
direct manifest-to-device composition.
That future conversion must validate every entry against the routed-expert
range and reject duplicate experts within each token row before narrowing.
The operation plan distinguishes attention-block and feed-forward-block mHC
control, pre-reduction, and post-combination phases, plus head reduction. Block
control generation consumes BF16-boundary activations with F32 weights and
produces F32 controls; reduction and combination return to the BF16 activation
boundary.

The generic model numeric schema is intentionally not forced to hold the
physical tensor list: its 65,536-tensor ceiling is below every DeepSeek-V4
profile, and its storage vocabulary cannot express UE8M0 block scales or the
shared canonical FP4 codebook. This family plan therefore grants no parsing,
dequantization, materialization, device, kernel, KV-cache, or scheduling
authority. `require_physical_materialization` always returns a typed failure.
