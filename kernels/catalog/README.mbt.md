# Exact kernel catalog

`kernels/catalog` resolves each semantic model operation to one exact startup
implementation for one device target. A content-addressed implementation names
a CUDA module digest plus a stable kernel-family identity. In legacy catalog
v1/v3, profile-specific entry points are selected later by
`kernels/launch_contract`.

The implicit cuBLASLt BF16 implementation is legal only for
`OutputProjection` and `LanguageModelHead`, whose current semantic layouts are
representable by the narrow single-GEMM device ABI. `QkvProjection` has three
weights and a packed Q/K/V output contract, so it requires an AOT family until
an explicit multi-descriptor vendor ABI exists. Silent decomposition or layout
fallback is forbidden. A nonempty vendor workspace requires the ABI's exact
256-byte alignment; incompatible catalog entries fail during admission.

Catalog v1 remains the stateless full-context contract. Catalog v3 is a
separate, fail-closed paged-KV contract: every entry records its exact semantic
version, optional decoder-layer identity, ordered live `OperationRuntimeInput`
roles, persistent-state effect, and required semantic KV layout. A v1 entry
cannot appear in a v3 catalog, and neither catalog version resolves a model
with the other execution mode.

Catalog v4 extends the paged-KV semantic contract with an exact
`OperationExecutionDigest` and catalog-owned `AotKernelEntryPoint` for every
entry. Numeric identity participates in semantic matching before device target
or implementation compatibility, so a target-compatible BF16, FP8, or I8
entry cannot substitute for another numeric contract. Reusing one AOT family
across different operation-execution digests is invalid. The entry point must
belong to the exact `ContentAddressedAot` family; vendor-library algorithms and
legacy entries are not legal in a v4 catalog.

V4 resolution returns only inert catalog evidence. It opens no artifact or
module and constructs no launch contract. `kernels/launch_contract` does not
select or replace v4 entry points: a future v4 launch path must consume and
reauthenticate the exact catalog-owned entry point and numeric digest already
retained by `KernelBinding`.

All validated catalog identities, entries, targets, execution semantics,
bindings, and resolved aggregates are opaque outside this package. External
callers can use their checked constructors and read-only accessors, but cannot
forge raw digests, IDs, workspace facts, catalog entries, bindings, or resolved
results. Only catalog resolution mints `KernelBinding` and
`ResolvedKernelCatalog` values.

Paged resolution additionally consumes the canonical `DeviceKvLayout`.
Catalog entries bind the layout version and tokens per page because both affect
AOT indexing. Total physical page count is deliberately not a kernel-family
specialization; it only sizes the later Key and Value component spans. Kernel
families may therefore be reused across decoder layers while each entry and
binding retains its exact `KvLayerId`.

Catalog v3 still selects only inert content-addressed AOT identities. It does
not admit the fixed-row vendor implementation because every paged operation
consumes live `StepCounts`. It does not claim that a device executor, CUDA
argument binder, or production paged kernel exists, and it has no JIT or
fallback path.
