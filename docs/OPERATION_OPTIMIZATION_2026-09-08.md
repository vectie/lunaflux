# Operation optimization follow-up

This work follows the measured gaps in `OPERATION_PROFILE_2026-09-08.md`.
Changes are evaluated independently before a combined Qwen serving comparison.
Compiler policies remain shape/capability based; CUDA mapping stays in lowering.

## Residual launch bounds

Production residual/RMSNorm V2 is row-local: one workgroup writes one token's
residual and normalized values. Previously its prepared graph retained the
1024-token grid at C1/C8. Startup planning now carries the live-token bucket
bound for this admitted ABI into ordinary, wide-prefill, partitioned-prefill,
and split-decode graph variants. Unknown/diagnostic ABIs retain fixed geometry.
No arithmetic, synchronization, kernel arguments, or live-row memory effects
change. The full-envelope fallback is preserved.

The new regressions cover C1/C8 through the full envelope, ABI exclusion,
undersized source envelopes, and rule remapping across attention expansion.
The device-step native suite passes 174/174. Physical performance is pending
the combined worker build; no speedup is claimed from the CPU tests.
