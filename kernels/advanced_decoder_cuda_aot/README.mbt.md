# Advanced decoder CUDA AOT foundations

This offline-only package binds advanced-decoder requirements to deterministic
CUDA source from the family-neutral decoder foundation and projection renderers
and emits canonical recipes for exact BF16 token embedding, final RMSNorm, and
the terminal language-model head. It also owns the family-neutral F32 expert
score transform source because that operation is already part of the common
advanced-decoder vocabulary. It consumes only family-neutral kernel
requirements, including explicit geometry, final-norm epsilon, score function,
and routed-expert width.
The paired Q/K RMSNorm foundation candidate binds the independent query-rank
and key-value-head widths to a family-neutral two-stream renderer. Both streams
use ordered F32 square-sum and scaling arithmetic with a single BF16-RNE output
round, while sharing only epsilon and the authenticated live-row prefix.
The preceding Query-A candidate reuses the family-neutral block-E4M3/UE8M0
projection source. It binds BF16 activations, replicated row-major E4M3 codes,
one raw UE8M0 byte per 128x128 weight block, and BF16 rank output. Query-B and
the intervening key/value projection remain separate operations.

Candidates bind model identity, complete geometry, requirement ordinal,
compiler policy, device target, maximum token profile, launch dimensions, and
exact operand byte contracts. They deliberately contain no compiled module
digest, catalog family, loader, execution authority, or qualification claim.
The language-model head binds row-major `[vocabulary,hidden]` BF16 weights,
ordered FP32 multiply/add accumulation without bias, live-token counts, and one
BF16 round at output.
The expert-score candidate accepts finite row-major router logits, applies the
reference's stable piecewise `sqrt(softplus(x))` formula in F32, and writes F32
scores for the live `counts[3]` row prefix. Its exact CUDA toolchain remains a
recipe input because transcendental results are not claimed bit-identical.
The token-hash candidate performs a bit-exact I32 lookup from the authenticated
row-major `[vocabulary,experts_per_token]` table. It binds the hash-layer count,
vocabulary, selected and routed expert counts, live-row capacity, counts-first
ABI, compiler, and target; invalid token IDs or table entries produce `-1`.
Family adapters must separately authenticate and perform any checkpoint-storage
conversion into this I32 ABI; the candidate does not grant such conversion.
The conversion boundary must validate the complete table's expert range and
per-row uniqueness before narrowing from checkpoint I64 storage.
The deterministic compressed-index candidate is a separate weight-free I32
operation. It consumes step counts, per-row absolute query positions, and
per-row cache offsets, then emits an exact per-row selected count and a dense
row-major complete-slot prefix with deterministic `-1` padding. The baked
maximum slot width is `floor(maximum_position_count / compression_ratio)`.
It owns neither compressed attention nor cache allocation/offset production.
The following deterministic join candidate makes the official two window
branches explicit with per-row I32 position, sequence-length, and mode inputs.
It preserves contiguous prefill padding or decode ring order, then appends the
producer's compressed I32 row literally, including `-1` padding. It performs
no sorting or deduplication and owns neither compressed-index production nor
K/V preparation.
The next non-overlapping shared-K/V preparation candidate consumes borrowed
ordinary and compressed BF16 segments. Prefill emits the literal full-sequence
then compressed-prefix layout; decode preserves physical circular-window slot
order then appends the completed compressed-cache prefix. It owns no learned
compressor projection/state or live cache and performs bit-exact BF16 copies.
Selected sparse attention is a separate eight-operand correctness candidate:
counts, BF16 query rows, I32 query-to-batch ids, borrowed BF16 shared K/V,
I32 cache lengths, F32 per-head sinks, already joined I32 selected indices,
and BF16 output. The sink contributes only to the stable-softmax denominator.
The following three-operand inverse-RoPE candidate accepts counts, absolute I32
positions, and one in-place complete BF16 attention head. It preserves the
non-RoPE prefix bit-exactly and applies conjugated scaled-YaRN rotation only to
the 64-wide suffix before output projection. Cache ownership remains a gap.
The scaled-rotary candidate covers only compressed YaRN rotation of separate
query and key/value RoPE suffix buffers. Its six operands are step counts,
absolute I32 positions, query input/output, and key/value input/output. It fixes
adjacent-pair interleave, a 64-wide RoPE dimension, unit-magnitude F32
`cosf`/`sinf` rotation, and BF16 RNE output. Plain-theta sliding-layer RoPE and
all non-RoPE head components remain outside this candidate.
The mHC block-control candidate binds a bounded BF16 stream input, separate F32
function/base/scale tensors, and separate row-major F32 pre, post, and
destination-major combination outputs. One thread owns each live row and fixes
normalized projection, sigmoid controls, stable row softmax, and the exact
alternating-normalization iteration count. Compiler-specific expf and sqrtf
results remain unqualified.
The following pre-reduction candidate consumes the BF16 stream state and F32
pre controls through directional operands, performs the stream-width reduction
in ascending order for every live row and hidden column, and rounds once to a
BF16 output. Post-combination and head reduction remain separate gaps.
The post-combination candidate binds a BF16 branch, BF16 residual stream state,
F32 post controls, destination-major F32 combination controls, and a BF16
combined stream output. It fixes initial post-times-branch arithmetic followed
by ascending-source accumulation and one output round.
The head-reduction candidate binds BF16 stream state, F32 function/base/scalar
scale tensors, and a BF16 reduced output. It fixes flattened-row normalization,
sigmoid control generation, ascending-stream reduction, and one BF16 output
round. CUDA `expf` and `sqrtf` remain compiler-bound numeric qualification gaps.
No-auxiliary-loss routing is represented as two distinct candidates. Stable
biased top-k consumes original F32 scores plus the F32 selection bias and emits
I32 indices, using lower expert id for exact ties. Selected-weight finalization
then gathers the original unbiased scores using either top-k or token-hash
indices, performs ordered F32 normalization when configured, and applies the
authenticated routing scale. PyTorch `topk` does not guarantee a stable tied
index order, so tied official-reference parity remains an explicit physical
qualification limitation rather than a bindability claim.
Remaining low-rank projections, sparse-attention preparation, expert execution, auxiliary prediction,
offline compilation, physical correctness, and benchmark promotion remain
separate gates.
The versioned output-A candidate consumes the officially converted BF16
group-major weights and performs only the first group-local projection. The
following output-B candidate separately consumes row-parallel E4M3/UE8M0
weights and emits only local F32 partial sums; collective reduction remains a
downstream gap.
