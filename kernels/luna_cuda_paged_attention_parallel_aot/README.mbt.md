# Parallel paged-attention CUDA AOT candidate

This native-only Phase 5 package derives one block-parallel BF16 paged-
attention candidate from an independently admitted serial reference lowering.
It binds the exact target, shape, paged-KV layout, operands, strict compiler
policy, a canonical LunaTile program/AOT plan, a 128-thread launch, bounded
dynamic shared memory, and a separately named tree-reduction tolerance policy.

The generated kernel assigns one block to each query-token/query-head pair.
Threads cooperate on K/V writes, QK dot products, maximum and denominator
reductions, and output components. Scores are computed once into bounded shared
memory rather than recomputing each dot for every output component. Current-
chunk K/V remains sourced from the QKV activation, so the candidate does not
depend on cross-block synchronization.

The serial `luna_cuda_paged_attention_aot` lowering remains an explicit,
separate fallback. Candidate source and recipe use a different symbol,
numerical-policy name, digests, entry-point ID, and launch geometry.

`admit_paged_attention_parallel_promotion_evidence` checks only bounded raw
offline claims: ordered differential/canary trials must satisfy the declared
tolerance and win each declared microbenchmark shape, while end-to-end trials
must preserve token agreement and the configured non-regression bound. The
candidate-only release join requires that evidence plus the exact compiled
entry point and source digest. Both candidate and joined result remain
`manifest_bindable=false`; neither grants runtime, loader, device, execution,
deployment, readiness, or external approval authority.

The checked-in tests prove deterministic lowering, scalar differential
agreement, dispatch-canary execution, hostile bounds, static resource identity,
and promotion-gate behavior. They are software tests, not a physical CUDA
benchmark, and this package makes no performance-win claim until the required
device measurements are independently produced and approved.
