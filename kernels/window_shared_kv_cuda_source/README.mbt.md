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
