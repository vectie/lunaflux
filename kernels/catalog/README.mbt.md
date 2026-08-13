# Exact kernel catalog

`kernels/catalog` resolves each semantic model operation to one exact startup
implementation for one device target. A content-addressed implementation names
a CUDA module digest plus a stable kernel-family identity; profile-specific
entry points are selected later by `kernels/launch_contract`.

The implicit cuBLASLt BF16 implementation is legal only for
`OutputProjection` and `LanguageModelHead`, whose current semantic layouts are
representable by the narrow single-GEMM device ABI. `QkvProjection` has three
weights and a packed Q/K/V output contract, so it requires an AOT family until
an explicit multi-descriptor vendor ABI exists. Silent decomposition or layout
fallback is forbidden. A nonempty vendor workspace requires the ABI's exact
256-byte alignment; incompatible catalog entries fail during admission.

Catalog v1 remains the stateless full-context contract. Catalog v2 is a
separate, fail-closed paged-KV contract: every entry records its exact semantic
version, optional decoder-layer identity, ordered live `OperationRuntimeInput`
roles, persistent-state effect, and required semantic KV layout. A v1 entry
cannot appear in a v2 catalog, and neither catalog version resolves a model
with the other execution mode.

Paged resolution additionally consumes the canonical `DeviceKvLayout`.
Catalog entries bind the layout version and tokens per page because both affect
AOT indexing. Total physical page count is deliberately not a kernel-family
specialization; it only sizes the later Key and Value component spans. Kernel
families may therefore be reused across decoder layers while each entry and
binding retains its exact `KvLayerId`.

Catalog v2 still selects only inert content-addressed AOT identities. It does
not admit the fixed-row vendor implementation because every paged operation
consumes live `StepCounts`. It does not claim that a device executor, CUDA
argument binder, or production paged kernel exists, and it has no JIT or
fallback path.
