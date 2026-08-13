# Execution manifest file admission

This package synchronously admits one canonical
`DenseLlamaPagedAotV1` execution manifest through a caller-owned
`ApprovedRoot` and an independently constructed `ApprovedRelativeLocator`.
The expected `ExecutionManifestDigest` is a separate lowercase SHA-256 value;
an immutable same-handle snapshot is closed successfully and hashed before
parsing or publishing any admission.

The v1 document contains only implementation claims: exact model identity,
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

Canonical v1 intentionally supports only the current Llama recipe with
`max_batch_rows == 1`. Multi-row execution requires a model-plan identity and
execution-manifest schema revision; a manifest cannot specialize around that
boundary. All filesystem and lower-layer semantic failures are mapped to
payload-safe file classes or admission stages, without retaining a path,
symbol, JSON value, descriptor, errno, or lower-layer error payload.

`PagedExecutionAdmission` is inert. It retains the exact weight layout and the
eight typed execution inputs
needed for device-worker/executor admission and exposes focused getters. It
contains no filesystem alias or native/device resource. The immutable admitted
artifact bundle, including immutable module bytes, is intentionally available
to startup code.
