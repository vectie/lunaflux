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

The compact alternative uses PackedCheckpoint: bind named payload, block-scale
and optional scalar-scale tensors to a shared Representation, then
transfer_weights streams their original bytes into caller-owned buffers.
One scratch buffer is reused across all weights; weights are not expanded to
BF16. The binding exposes scale offsets for device operand decoding. Four-bit
weights retain low-column/low-nibble packing and scale planes remain row-major.

This is a startup data path, not a serving kernel or an activation-quantization
policy. GLM's NVFP4 and DeepSeek's FP4/FP8 expert adapters use it for per-device partitions;
existing BF16 startup expansion remains available for other weight plans.

transfer_weights_to also streams multiple projections into one bank at explicit
destination offsets. It shares PackedBufferLayout with the compiler, so the
global-scale padding seen by CUDA matches the bytes populated by the uploader.
Packed E2M1 checkpoint bytes accept U8 or I8 storage tags; UE8M0 scale bytes
accept U8 or F8_E8M0. These are explicit raw-byte aliases, not signed-value
conversion, and exact physical shapes/byte counts still apply.
