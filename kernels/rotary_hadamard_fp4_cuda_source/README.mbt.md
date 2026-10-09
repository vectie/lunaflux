# Rotary / Hadamard / FP4 simulation

Three effects lower the immutable `RotaryHadamardFp4Precision` numerical plan:
suffix RoPE with BF16 publication, normalized Sylvester Hadamard with BF16
publication, then whole-vector blockwise E2M1 quantize/dequantize back to BF16.
The FP4 scale is the smallest power of two not below `amax / 6`, with the
reference floor `6 * 2^-126`. Midpoint ties choose the even E2M1 code.

Hadamard stages use separate shared input/output arrays and a barrier before
swapping ownership; no thread overwrites another thread's unread prior stage.
The generic rotary renderer is shared with ordinary compressed attention.
This correctness-first lowering is not a performance-optimized Hadamard kernel,
and simulated BF16 output must not be described as packed FP4 cache storage.

Native tests cover plan budgets, source stages and model-adapter selection.
The independent `sm121` fixture covers positions 0/3/1023, live rows 0/1/2/3,
replay and even-code midpoint ties. Final numerical error is zero after the CPU
oracle preserves reciprocal-frequency/multiply-position F32 operation order.
The Hadamard scalar is computed immutably rather than replaced by approximate
device `rsqrt`. This component result does not prove real-checkpoint accuracy.
Memcheck/full leak, racecheck and synccheck report zero errors/hazards/leaks
for that final fixture; failed earlier probes remain preserved.
