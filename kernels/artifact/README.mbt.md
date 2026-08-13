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
