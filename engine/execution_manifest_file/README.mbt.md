# Execution manifest file admission

This package synchronously admits one canonical paged-AOT schema-v2 execution
manifest through a caller-owned
`ApprovedRoot` and an independently constructed `ApprovedRelativeLocator`.
The expected `ExecutionManifestDigest` is a separate lowercase SHA-256 value;
an immutable same-handle snapshot is closed successfully and hashed before
parsing or publishing any admission.

The v2 document contains only implementation claims: exact model identity,
device target, catalog version 3, one profile identifier, lexically ordered
content-addressed module/export declarations, and one operation declaration in
canonical model-plan order. Each operation selects a module family, entry
point, bounded CUDA launch dimensions, and exact workspace requirement.
Duplicate or unknown fields, non-canonical ordering, unused declarations,
invalid symbols, and mismatched identity/target/version fail closed.

Semantic execution facts are never deserialized. The caller supplies an
already-admitted paged `ModelPlan`, `DeviceWeightLayout`, `DeviceKvLayout`,
`DeviceTarget`, and `DeviceStepLimits`. Admission derives catalog-v3 execution
semantics, resolves the catalog, builds the static and memory plans, derives
the exact full-graph operand roles/sizes/alignments, admits a single
`PagedKernelProfile`, proves `FullGraph` contracts, admits the artifact source,
loads immutable module snapshots through the same approved root, and finally
admits the device-step blueprint.

Canonical v2 admits the exact batch-row envelope already bound into the typed
model-plan identity. `DeviceStepLimits.max_rows` must equal that plan ceiling;
token and page envelopes are checked with widened row-multiplied arithmetic
against sequence geometry and physical page capacity. The JSON cannot override
or specialize any of those typed bounds. All filesystem and lower-layer semantic failures are mapped to
payload-safe file classes or admission stages, without retaining a path,
symbol, JSON value, descriptor, errno, or lower-layer error payload.

`load_tensor_parallel` reuses that same snapshot, parser, manifest claims, and
full semantic catalog derivation for an exact rank-local execution admission.
It cross-authenticates model generation, the sharded file inspection, generic
rank plan, device plan, device ordinal, target, and local KV binding before deriving one profile,
the rank-local AOT contracts, exact artifact bundle, and immutable physical
execution plan. The returned admission owns no filesystem or device authority
and never constructs a full-model weight layout. Schema v3 Luna specialization
is rejected until an exact tensor-parallel specialization contract exists.

`PagedExecutionAdmission` is inert. It retains the exact weight layout and the
eight typed execution inputs
needed for device-worker/executor admission and exposes focused getters. It
contains no filesystem alias or native/device resource. The immutable admitted
artifact bundle, including immutable module bytes, is intentionally available
to startup code.
