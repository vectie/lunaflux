# Luna attention strategy

Backend-neutral attention planning separates prefill query tiling from decode
key/value splitting and online-softmax merge. Hardware backends own subgroup,
shared-memory, vector-width, and matrix-instruction lowering.

The compiler API represents each serving bucket as an
`AttentionCompileProblem`. It generates a bounded set of portable tile
candidates, prunes them against abstract backend capabilities, and selects one
with either immutable offline measurements or a deterministic static cost
model. The selected plan fixes query and key/value tile geometry, arithmetic
class, memory pipeline, subgroup count, pipeline depth, vector width, and
shared-memory demand before device lowering. Candidate generation, validation,
and autotune scans are startup/compiler work and never execute in the token
step.

`compile_attention_tile_candidate_plan` lets the higher functional compiler
reify only members of this bounded set. It is the typed bridge for
post-rewrite resource feedback and cannot admit arbitrary candidates or
manufacture an autotune observation.

Decode workspace planning divides the admitted context into ordered,
page-aligned, nonempty partitions and defines one bounded F32 layout for local
maximums, denominators, and numerator vectors. Split count is clamped to the
available page count. One split covers the complete context and is equivalent
to the ordinary online-softmax reduction; multiple splits merge in ascending
partition order. No CUDA, warp, or vendor vocabulary enters this package.
