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
