# Bounded precision startup materialization

Consumes an explicit `StartupExpandBf16` weight plan. Small immutable packed
tensors can be converted into caller-owned buffers; large tensors use bounded
streaming readers and an upload sink. File authentication, arena ownership and
deterministic device release stay with the loader, not the scalar codec.

This adapter is a startup effect, not a token-step fallback. Host scratch and
resident BF16 memory have separate explicit limits. Compact tile execution uses
the other precision route and does not require this expansion.

`stream_bf16_weights` interprets the complete immutable `WeightSchema`. It
preflights every placement and global scale before effects, uses tensor ordinal
readers, and writes the planned aligned offsets. The caller keeps the arena
unpublished until all reads/conversions/uploads succeed and releases it on error.

See [precision layer and integration status](../../docs/PRECISION_IR_2026-10-04.md).
