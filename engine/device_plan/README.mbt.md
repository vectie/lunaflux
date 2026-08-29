# Static device planning

`engine/device_plan` is the immutable startup bridge between a validated
semantic model plan, a deterministic device-weight layout, and an exact
resolved kernel catalog. Kernel selection must name the exact execution plan.
Legacy catalog v1/v3 plans preserve content-digest reuse across full-context
and paged plans. Catalog v4 instead requires the complete model identity, exact
numeric-schema digest, exact per-operation execution digest, v4 semantic
version, and the catalog-resolved content-addressed AOT entry point.

Before resolving any semantic input, the builder revalidates the complete
opaque layout in exact numeric tensor-table order. Parameter, scale,
zero-point, and codebook regions must have canonical aligned offsets and exact
`numeric_contract.storage_byte_length` sizes; their checked materialized sum
and terminal extent must match the advertised arena. The resulting static plan
retains that complete authenticated layout, including metadata tensors not
directly referenced by semantic operations, and exposes only read-only
accessors. It also retains the model numeric-schema digest and exact execution
digest for every operation.

Planning is bounded to 4,096 operations, 64 inputs per operation, and 8,192
inputs in total. Unsupported or inconsistent startup data fails with a typed
payload-safe error before an executor is created.

This package is planning only. It does not own a CUDA context or allocation,
load AOT artifacts, invoke cuBLASLt, launch kernels, allocate activation or KV
arenas, schedule requests, or prove physical-GPU correctness. In particular,
constructing `StaticDevicePlan` is not evidence that production CUDA execution
is available. The legacy stateless `device_profile` bridge rejects catalog v4;
numeric-aware profile and launch preparation remain later work.
