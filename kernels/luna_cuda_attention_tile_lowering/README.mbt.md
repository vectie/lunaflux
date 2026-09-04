# Luna CUDA attention tile lowering

This is the NVIDIA-specific lowering boundary for the functional attention tile
compiler. It maps generic arithmetic and memory classes to CUDA primitives and
derives launch geometry from the immutable schedule. CUDA matrix instructions,
subgroup width, asynchronous copy, `sm_*` identity, and launch limits exist
only here and must never flow back into model, scheduler, strategy, semantic IR,
or generic schedule packages.

The functional schedule can request one paged K/V address calculation per
logical key row. CUDA realizes that portable reuse decision with subgroup
broadcast across the vector fragments that consume the row; no CUDA primitive
is reflected back into the optimizer or schedule vocabulary.
