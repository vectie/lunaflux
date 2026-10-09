# DeepSeek V4 CUDA AOT integration evidence

This test-only package joins all five closed official DeepSeek V4 semantic
profiles to the family-neutral advanced-decoder I32 token-hash, stable biased
top-k, selected-weight finalization, Q/K RMSNorm, language-model-head, and
expert-score lowerers plus the first exact mHC block-control lowerer. It proves
the width-4, 20-iteration mHC geometry and separate F32 control buffers, exact
requirement ordinals, the
Flash/Pro 1024/1536 query ranks, the shared 512-wide key-value head, terminal
hidden widths, the 129280-token vocabulary, I32 hash-table geometry, BF16
tensor geometry, and the 256/384 routed-expert F32 score geometry. It also checks live-count ABI
alignment, identity binding, distinct profile recipes, hostile adjacent-
operation rejection, and non-bindable status without importing DeepSeek types
into kernel packages. All five profiles also lower the exact ratio-128
deterministic complete-slot index operation. Its explicit I32 positions,
cache-offsets, selected-counts, and dense `-1`-padded slot matrix remain
separate from compressed-attention and KV ownership.
All profiles also bind the literal window-first join. Prefill uses causal
contiguous indices with official padding, decode uses the circular 128-slot
ring order, and compressed indices retain their original order and `-1`
padding without sorting or deduplication.
All five profiles also lower selected sparse attention with 64/128 query
heads, a 512-wide shared K/V vector, 128 window slots plus 8192 compressed
slots, and a denominator-only F32 sink. The joined-index producer,
non-overlapping K/V preparation, inverse RoPE, and cache ownership remain
separate typed gaps.
The query requirement is split into query-A at ordinal 1, paired Q/K RMSNorm at
ordinal 3, and query-B at ordinal 4. Query-A weights are `[rank,hidden]` and
query-B weights are `[heads*512,rank]`; both use E4M3 parameters with UE8M0
128x128 scale grids. Query-A now has a replicated correctness candidate and a
family-owned inert parameter join to the exact checkpoint layout. Query-B and
key/value projection remain typed gaps; no BF16/scalar-scale substitute is
admitted.
The next mHC phase binds BF16 `[rows,4,hidden]` stream state and F32
`[rows,4]` pre controls to a BF16 `[rows,hidden]` reduction output with one
rounding boundary.
Post-combination then binds BF16 branch/residual inputs, F32 post and
destination-major combination controls, and the width-4 BF16 stream output.
Head reduction binds width-4 BF16 stream state, F32 `[4,4*hidden]` function
weights, F32 `[4]` base, scalar F32 scale, and BF16 `[rows,hidden]` output.
All mHC candidates remain non-bindable. Complete artifact admission advances
through Query-A, then fails at key/value projection ordinal 2 rather than
treating placeholder dense operands as authority.
Post-attention output projection is split into group-local output-A and
row-parallel output-B. Output-A uses the official converter's BF16
`[groups,1024,4096]` layout and has a correctness source candidate; output-B
has a separate row-parallel local-partial E4M3/UE8M0 candidate while the
collective reduction remains a typed gap.

Top-k and finalization remain separate because the first three hash-routed
layers obtain indices from the token table while all paths still gather
original unbiased scores, normalize, and scale selected weights. LunaFlux fixes
lower expert id for exact score ties; official PyTorch `topk` tie parity remains
unqualified because that API does not promise stable tied indices.

The official checkpoint table is I64 while the reference runtime lookup is
I32. A separate DeepSeek host-materialization package now performs the
authenticated checked I64-to-I32 conversion; these tests cover only the I32
compute candidate and do not compose that sidecar into device memory or claim
direct weight-manifest binding, runtime execution, or qualification.
