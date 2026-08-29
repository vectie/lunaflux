# FP8 dynamic-scale launch ABI

This host-only package joins an immutable model plan, an admitted FP8 kernel
manifest, and an already-admitted stateless or paged launch-contract set. It
derives model, schema, manifest, target, profile, operation, execution, module,
and entry-point identities from those authorities. Callers provide only the
exact live execution shape and opaque storage regions for manifest-owned
scalar weight scales, the dynamic activation-scale cell, and any declared
kernel workspace.

Every region binds a storage identity, total capacity, offset, length, and
power-of-two alignment. Admission rejects overflow, out-of-bounds placement,
capacity disagreement, partial overlap, unsupported scaling semantics,
profile/shape substitution, stale schema or manifest evidence, and mismatched
kernel identity. Stateless shapes select an exact profile explicitly; paged
row storage is detached from the caller and is checked against the admitted
profile and device KV page geometry.

`Fp8DynamicLaunchOwner` supplies a single shared acquire/use/finish/release
authority with epoch fencing. Foreign, stale, replayed, and concurrent leases
fail closed. Terminal close is deterministic and idempotent and invalidates
every outstanding lease.

## Version 2 whole-Workspace ABI

`admit_staged_dynamic_activation_scale_launch_v2` accepts one caller-owned
`Workspace` region rather than independently supplied activation-scale cells.
The operation execution policy and kind come only from the authenticated model
plan and manifest. V2 accepts only catalog-v4 paged launch contracts: legacy
stateless and paged contracts cannot authenticate weight-scale operands and are
rejected. The selected contract must contain every manifest-owned
`WeightScaleInput` immediately after its matching `WeightInput`, then end in
exactly one `Workspace` operand with alignment four. Simple QKV, output, and
language-model-head projections own four bytes, while a compound gated MLP owns
eight bytes.

The package derives ordered, typed four-byte subregions without accepting
caller offsets or stage claims. A simple projection exposes only
`external_operation_input_v1`; a compound gated MLP additionally exposes
`post_silu_gate_up_product_v1`. Both cells retain the whole Workspace storage
identity and capacity, and their offsets are fixed at `workspace.offset` and
`workspace.offset + 4`. Weight-scale regions must be disjoint from the whole
Workspace.

Version 2 uses the independent canonical domain
`lunaflux.fp8.staged-dynamic-launch-abi.v2`. It binds the policy, operation
kind, whole Workspace, ordered stage identities and derived subregions in
addition to the existing model, schema, manifest, profile, exact-shape,
module, entry-point, and weight-scale identities. It also retains and binds the
catalog-v4 source version, exact grid/block/shared-memory dimensions, and every
ordered raw operand role, reference ordinal, byte count, and alignment.
Version-1 canonical bytes, digest domain, admission API, and ownership API are
unchanged.

This package opens no CUDA module or allocation, launches no kernel, and owns
no executor or service route. Its ABI is necessary host admission evidence,
not proof of a real AOT module, physical numerics, memory savings, performance,
or production readiness.
