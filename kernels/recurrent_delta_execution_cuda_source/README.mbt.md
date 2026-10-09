# Recurrent delta device lowering

This package lowers the generic normalized causal delta precision plan, not a
GLM-specific model graph. Q/K/V, beta and output are BF16; log decay and persistent
state are F32. Independent key and value dimensions are supported up to 256.

A CTA owns one sequence/head. A value lane folds keys in their specified order:
decay state, read with normalized K, calculate `(V - read) * beta`, perform the
rank-one update, and project with normalized, scaled Q. Token order stays causal.
The request cache is accessed before and after the frame, not published after
every token. Lane-private state can spill; it is not a guarantee of register-only
execution.

The launch uses CSR sequence offsets and explicit persistent slot/reset arrays.
Slots are unique within a frame by scheduler construction. Normal batch reordering
does not move cache contents; a reset starts a new request in that slot. Idle and
inactive rows/slots remain untouched. The kernel consumes prepared inputs without
qualification scans or host state readback in the token loop.

The old serial source is imported only by export tests as a numerical reference.
`physical_probe.cu` also checks an independent double-precision recurrence across
slot continuation, reorder, reset, unequal dimensions and idle frames. Its paired
timings compare those two component schedules, not whole-model inference engines.
