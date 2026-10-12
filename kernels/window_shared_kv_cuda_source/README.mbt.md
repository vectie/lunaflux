# Retained shared-KV window lowering

This correctness-oriented CUDA lowering consumes the model-neutral
`WindowSharedKvPrecision` plan. One request owns a BF16 ring; query/current-KV
frames remain separate until every row's causal sink attention finishes.
This permits chunked prefill longer than the ring without overwriting the
history needed by early rows. Reserve checks contiguous absolute positions,
context capacity and a sticky error; copy retains only the newest word-exact
KV span; publication advances history only after those effects finish.

Scores are computed once per selected key using ordered F32 dot products.
The learned F32 sink contributes only to the denominator, and output rounds
once to BF16. This is not a tuned long-prefill kernel. It implements no
compressed cache, learned indexer or inverse rotary/output projection.

An opt-in `AttentionAccumulationLaw::TiledFloat32` policy now separates
permitted F32 reassociation from the probability-storage law. Its CUDA
lowering uses BF16 matrix-MMA, a parallel denominator reduction and explicit
row-owned F32 fragment rescaling. Query heads share KV operand tiles. The
16-head physical ownership and warp instructions remain in this CUDA package,
not model or scheduler semantics. The source retains the policy, and launch
metadata carries its grouped grid and 256-thread block into prepared frames.

The first lowering requires BF16-capable WMMA (SM80+), head widths divisible by
16 up to 512, and online BF16 probability tiles of 32/64/128 keys. Head and
column tails are zero-padded only inside physical matrix tiles; real output
geometry is unchanged. Unsupported combinations return `NoImplementation`,
never silently substitute ordered arithmetic. Shared staging is bounded by
the physical tile, not the complete context. Default ordered source remains
byte-identical and production selection is unchanged.

On captured 144-row DeepSeek operands, all 4,718,592 candidate BF16 outputs
match the installed original attention bitwise. Prefill and incremental ring
execution agree, including an 18-head/80-column tail control. Memcheck/leak,
racecheck and synccheck pass; this is local attention evidence, not complete
decoder or full-model parity. See
[the matched original report](../../docs/DEEPSEEK_ORIGINAL_STREAM_REPLAY_2026-10-11.md).
