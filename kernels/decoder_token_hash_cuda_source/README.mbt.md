# Decoder token-hash CUDA source

This family-neutral package renders deterministic I32 row-major token-table
lookup CUDA source from exact vocabulary, selected-expert, routed-expert, and
maximum-row geometry. The counts-first ABI bounds the live `counts[3]` prefix;
invalid token IDs or expert entries produce the fail-closed `-1` sentinel.

The result is source text only. It contains no model identity, compiler
invocation, artifact bytes, loader, runtime launch authority, or qualification
claim.
