# DeepSeek V4 kernel requirement projection

This startup-only adapter projects an immutable DeepSeek-V4 semantic plan into
the family-neutral advanced decoder AOT requirement set. It binds the verified
checkpoint content and exact advanced execution-plan digest before returning.
The query path is explicitly ordered as FP8 query-A, the existing Q/K RMSNorm,
then FP8 query-B. All five official profiles use finite-E4M3 parameters with
128x128 UE8M0 block scales, so neither projection is weakened to the existing
BF16 dense or scalar-F32-scale FP8 contracts.

It performs no artifact selection, kernel qualification, device work, KV-cache
management, scheduling, or execution.
