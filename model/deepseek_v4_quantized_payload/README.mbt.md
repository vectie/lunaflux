# DeepSeek V4 quantized host payload contract

This startup-only package admits exact bounded host byte layouts for the four
quantized storage roles already present in the DeepSeek V4 logical numeric
plan: finite E4M3 parameters with 128×128 scales, UE8M0 scale grids, packed
E2M1 expert parameters with 32-element row scales, and their UE8M0 grids.
Every contract binds the storage encoding, exact block shape, source tensor
logical shape, model content digest, byte count, and payload SHA-256.

Only finite E4M3FN scalar interpretation is established by captured LunaFlux
evidence, including the two non-finite codes. UE8M0 exponent bias and special
codes are not established. The captured FP4 evidence establishes two values
per byte and the canonical E2M1 codebook identity, but not low/high nibble
ordering, codebook entry ordering, or odd-column padding. Those paths return
typed unavailability instead of guessing. Consequently every scale-dependent
dequantization remains unavailable.

This package performs no device allocation, kernel selection, scheduling,
materialization, or production fallback.
