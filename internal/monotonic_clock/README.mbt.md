# Monotonic clock native boundary

This package is the sole owner of LunaFlux's monotonic-clock foreign
declaration. The C translation unit reads `CLOCK_MONOTONIC`, validates the
returned `timespec`, and converts it to `UInt64` milliseconds with an explicit
overflow check. It never reads civil or wall time.

Only primitive status and output values cross the ABI. The runtime facade maps
the private native error vocabulary into its own payload-safe public errors.
`monotonic_clock_probe.c` includes the production translation unit with a
deterministic clock-read substitution, so the exact-TU sanitizer gate covers
success, native failure, invalid nanoseconds, null output, and overflow without
adding test hooks or symbols to the production ABI.
