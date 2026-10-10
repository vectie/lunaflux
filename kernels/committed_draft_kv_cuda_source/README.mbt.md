# Committed-main plus noncausal draft attention

The shared immutable precision plan defines a retained BF16 main ring and a
separate draft read set. This AOT correctness lowering publishes main rows before
prediction attention, reads physical retained slots followed by all live draft
slots, and includes a learned F32 softmax sink. It never writes draft KV to the
main ring. Initial priming has no attention launch.

The CUDA score-storage limit belongs to this renderer, not semantic IR. This
first serial-score lowering prioritizes exact read sets and deterministic F32
arithmetic; it is not an optimized or benchmark-qualified attention kernel.
