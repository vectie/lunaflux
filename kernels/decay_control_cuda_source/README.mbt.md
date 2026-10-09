# Family-neutral decay-control CUDA source

This package renders a bounded pointwise correctness kernel for BF16 raw
forget logits, F32 component bias, and F32 per-head log rates. It performs the
declared F32 bias, exponent, rate multiplication, sigmoid, and negative safe-
bound multiplication in order and writes F32 log decay.

It owns no projection, recurrence, cache, compiler, artifact, qualification,
launch, or execution authority.
