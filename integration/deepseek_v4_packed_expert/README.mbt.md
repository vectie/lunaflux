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

The common `MoeProgram` queues automatic routing, routed experts, optional
rank-local F32 reduction, replicated shared work and finalization. The adapter
now supplies BF16 router weights and sqrt-softplus scores for both DeepSeek
routes: the first three layers preserve token-table expert order and gather
learned score weights; subsequent layers use bias-corrected top-k with unbiased
normalized weights. No bias is fabricated for token-hash layers.

`prepare_routed` borrows the existing startup-narrowed I32 token table and its
allocation-relative offset, so different layers can share a sidecar arena.
Neither table lookup nor selected score gathering copies routing to the host.
The model adapter does not introduce a second executor or a new transport.

All decoder layers, compressed-attention caches, mHC and DSpark iteration still
need to be connected into whole-model serving. These MoE features do not yet
establish complete model generation or throughput.
