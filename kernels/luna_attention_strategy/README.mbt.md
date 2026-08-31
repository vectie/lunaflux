# Luna attention strategy

Backend-neutral attention planning separates prefill query tiling from decode
key/value splitting and online-softmax merge. Hardware backends own subgroup,
shared-memory, vector-width, and matrix-instruction lowering.

Decode workspace planning divides the admitted context into ordered,
page-aligned, nonempty partitions and defines one bounded F32 layout for local
maximums, denominators, and numerator vectors. Split count is clamped to the
available page count. One split covers the complete context and is equivalent
to the ordinary online-softmax reduction; multiple splits merge in ascending
partition order. No CUDA, warp, or vendor vocabulary enters this package.
