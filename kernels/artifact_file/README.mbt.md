# Bounded AOT artifact-file admission

`kernels/artifact_file` admits production AOT CUDA artifacts through one
caller-owned `runtime/approved_fs.ApprovedRoot`. The pinned root remains open
and owned by the caller on success and failure. A lexical manifest identifier
and independently obtained SHA-256 select an exact JSON manifest that pins the
model content and plan identities, exact device target, catalog version,
module digests and relative identifiers, and stable family/entry-point
identities with bounded CUDA symbols.

The same parser and same-handle loader admit both stateless catalog-v1 launch
contracts and paged catalog-v3 launch contracts. The contract set supplies the
expected model, target, and catalog evidence; manifests cannot select or
downgrade that evidence.

Descriptor-relative `openat(O_NOFOLLOW)` traversal and final type checks belong
to `runtime/approved_fs`; this package does not concatenate or rediscover
paths. Module files are opened together and their per-file and aggregate sizes
are proven before any module-sized host allocation. Each module is then read by
one lifecycle-leased immutable-snapshot operation, checked against the
preflight size, and SHA-256 verified. Admission delegates required-module,
required-entry-point, symbol, and content semantics to `kernels/artifact`,
which independently rechecks content identity before publishing a bundle.

Execution-manifest reconstruction may supply an already admitted inert
`KernelArtifactManifest` directly to `load_admitted` or
`load_paged_kv_admitted`; this reuses the same bounded module opener and never
serializes or reparses a nested JSON artifact manifest. A deployment still
supplies the authority that the pinned root is approved and read-only. There is
no executable metadata, search path, environment lookup, cache, compiler, JIT,
or fallback channel.

`load_tensor_parallel_admitted` uses that same bounded same-handle module
loader, then authenticates exact modules and entry points against the supplied
generic rank-local tensor-parallel launch-contract set. It adds no discovery,
model-family branch, device authority, or native module loading.
