# Prefill score deforestation

The long-prefill investigation separates fixed 256-token forward granularity
from the cost of each attention fold. The first lowering correction realizes
the existing backend-neutral `FuseScoreTransform` region: compose scale and
causal mask with the softmax consumer instead of materializing another shared
score tensor. This is map/fold deforestation, not a model-specific heuristic.

The CUDA lowering already requires that region. Its terminal implementation
may use CUDA primitives, but model builders, scheduling, and the semantic
optimizer remain device-neutral. Keep the original matrix dot, softmax
reduction order, probability type, and explicit float multiplication rounding.
Do not introduce runtime tuning or change numerical tolerances.

Validation plan: affected compiler tests; paired old/new physical attention
probes at identical tile shapes with long and ragged contexts; sanitizer
checks; then isolated Qwen serving measurements. Until measured, this is not a
claimed end-to-end improvement or a solution to the separate chunking cost.
