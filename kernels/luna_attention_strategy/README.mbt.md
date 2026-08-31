# Luna attention strategy

Backend-neutral attention planning separates prefill query tiling from decode
key/value splitting and online-softmax merge. Hardware backends own subgroup,
shared-memory, vector-width, and matrix-instruction lowering.
