# Functional CUDA attention AOT compiler

This package closes the pure compilation path from an immutable attention
problem through strategy selection, LunaTile semantic IR, functional map/fold
scheduling, CUDA lowering, and deterministic source emission.

Toolchain execution and filesystem writes are intentionally outside the
compiler. The same functional prefix can feed a different device lowering.

The optional frontier entry point maps the generic compiler's non-dominated
kernel family to stable suffixed CUDA symbols. It does not choose runtime
buckets: that decision remains in the backend-neutral execution-graph layer,
while this package only performs terminal CUDA lowering and source emission.

Both request types accept immutable resource feedback through
`with_resource_feedback`. Backend tooling supplies facts from the exact target
and compile flags; this package invokes no compiler process or profiler. In
particular, a capped-register cubin and an uncapped cubin must not share a
measured latency/profile identity merely because their source is identical.
Query-owned and dense-current read contracts flow from the generic problem
through semantic identity, schedule, lowering, and source emission.

`enumerate_cuda_partitioned_attention_tiles` lowers the joint candidate/target
search into complete source pairs with distinct partial and merge symbols.
`select_measured_chain` compares those exact compiled pairs using the shared
continuation-aware selector and actual scratch/shared-memory requirements.
Only whole-chain observations participate, with at least three samples and
matching frontier, compilation and caller-supplied measurement scope. Isolated
partial timings and sums of isolated medians are not whole-chain timings.
Missing measurements do not become zero cost or a guessed fallback.

Scope must distinguish the physical device, compiler/toolchain and flags,
workload, launch mode and cache protocol. The source frontier alone does not
identify a cubin built with a different register ceiling. A selected source
pair is an offline choice, not runtime admission or permission to use a
different numerical law. Native resource/correctness checks and real dispatch
validation remain necessary before changing a serving route.
