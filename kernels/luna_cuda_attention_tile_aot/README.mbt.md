# Functional CUDA attention AOT compiler

This package closes the pure compilation path from an immutable attention
problem through strategy selection, LunaTile semantic IR, functional map/fold
scheduling, CUDA lowering, and deterministic source emission.

Toolchain execution and filesystem writes are intentionally outside the
compiler. The same functional prefix can feed a different device lowering.
