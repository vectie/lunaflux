# Content-addressed AOT artifact admission

`kernels/artifact` admits the exact CUDA modules and exported functions required
by an already admitted launch-contract set. Module bytes are SHA-256 verified,
defensively owned exactly once per digest, and shared by every required family
and profile-specific entry point in that module.

Entry points use stable integer identities during planning and dispatch. CUDA
symbol strings are bounded startup metadata only. Admission rejects missing,
duplicate, unreferenced, and invalid declarations before device module import.
Distinct stable entry-point identities may intentionally map to the same proven
CUDA symbol in one module. This package performs no filesystem lookup,
executable metadata evaluation, compilation, cache access, or runtime JIT.

`admit` preserves the catalog-v1 stateless path. `admit_paged_kv` accepts only
catalog-v3 paged-KV launch contracts and admits only the positioned-rotary and
paged-attention entry points those contracts reference. Repeated contracts may
share one stable entry-point identity across profiles or layers, and multiple
stable identities may intentionally name the same CUDA symbol. The returned
bundle is immutable artifact evidence, not an executor or full-graph support
claim.

`admit_paged_v4` accepts only an already admitted catalog-v4 full-graph launch
contract set. It retains required entry points in exact first-occurrence
profile/operation order and delegates to the same ownership, content-digest,
symbol, missing, duplicate, and unreferenced checks. It does not read an
artifact file, import a module, construct a blueprint, launch a kernel, or
claim physical device readiness. Until the shared legacy admission index is
replaced, this v4 path rejects more than 1,024 total profile/operation
contracts before allocating or scanning its required-entry table. The D3
artifact-v4 admission subphase owns this bounded-comparison debt. Replace the
shared index with linear lookup and remove the cap before D3 executor
integration begins; that executor integration is the latest removal phase.

`admit_tensor_parallel` accepts only an already admitted catalog-v3 rank-local
launch-contract set. It derives the unique entry-point list from every
profile/operation contract and delegates to the same module ownership, digest,
symbol, missing, duplicate, and unreferenced admission path. It introduces no
second artifact bundle, binary-semantics claim, filesystem path, device handle,
or execution authority.
