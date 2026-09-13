# Prefill composition integration

This work implements the five remaining composition seams, not a claim of
completed end-to-end performance qualification.

- Query-owned single-storage async K/V scheduling separates K-score readiness
  from V readiness. The previous two-storage candidate remains measurable, not
  the default merely because it uses async instructions.
- Warp-uniform interior/boundary specialization shares the maximum fold.
- Projection folds share immutable fragment lifetime lowering, including the
  sibling-reuse path. Wide row products retain a compact register lifetime.
- Offline attention observations are parsed in a backend-neutral package and
  bound to the exact frontier, device, toolchain, and workload vector. They
  choose one AOT variant; this is not per-request runtime autotuning.
- Optional version-1 attention metadata is constructed once per step in
  preallocated descriptor storage and shared across heads/layers. Bundle v5
  explicitly distinguishes metadata from legacy row offsets. Use the `.mbtx`
  fused builder for metadata-enabled recipes; do not use the legacy shell
  builder's intermediate bundle as a metadata runtime.

## Measurements so far

On the RTX 5060 Ti, `metadata-r1` compared with `activation-r3` using identical
inputs and strict arithmetic. With 1,528 **total** query tokens, eight rows and
no history, paired median kernel times were approximately 134 → 112 microseconds
for synchronous query-owned attention and 134 → 123 for single-storage async.
Single-row async remained slower (approximately 469 → 500 microseconds).
All 16 comparisons were bitwise equal, with an independent sampled scalar
referee, unchanged KV, and memory/race/synchronization sanitizer passes.

These timings exclude host metadata construction and upload. End-to-end testing
must account for those costs before enabling the metadata ABI by default.

The first fragment campaign covered 45 paired shape/family cases, all bitwise
equal, plus three sanitizer modes for each family. The sibling-reuse lookahead
variant regressed about 4% in the bare static configuration; the subsequent
compact lifetime policy and production-tuned configuration require retesting.
Bare static projection recipes are not the production tuning configuration.

Local full native suite: 3,737 passed. Further integration checks and measured
table ingestion are ongoing. No production deployment was performed.
