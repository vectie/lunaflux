# DeepSeek compact expert execution

The model adapter supplies checkpoint names, shape, clamp, precision and routing
order to the generic ExpertMlpPrecision plan. It does not introduce a second
executor, CUDA-specific model semantics, or token-path model-name switches.

The routed-expert chain is:

BF16 input → block128 FP8/UE8M0 → gate/up → F32 clamped SwiGLU and routing score
→ BF16 → block128 FP8/UE8M0 → down → BF16 → rank-local F32 sum.

amax is floored at 1e-4, and scales round upward to powers of two. Expert bank
upload uses one bounded reusable host scratch buffer. Allocations and release
remain caller-owned. The generic binder borrows two compact activation buffers
and a cleared device error word; it prepares all five AOT launches at startup.

This does not yet wire shared experts, global expert reduction, all decoder
layers, caches, mHC and DSpark iteration into whole-model serving.
