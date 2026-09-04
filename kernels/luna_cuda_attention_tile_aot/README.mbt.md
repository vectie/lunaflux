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
