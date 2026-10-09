# Full-head rotary and KV precision simulation

Consumes the immutable `RotaryKvPrecision` contract. Both operands retain
`[row, head, full_head_width]` storage; adjacent F32 rotary affects only the
suffix and rounds once to BF16. The non-rotary prefix is copied as raw words.
A second operation performs blockwise E4M3 encode/decode into BF16 for KV's
non-rotary prefix. Linear-F32 and power-of-two scale policies are explicit.
The cache remains BF16: this is QAT simulation, not a packed FP8 cache.

CUDA thread/reduction geometry lives here, not in the model or precision IR.
No dynamic compilation, cache commit or device submission occurs in this
renderer. `integration/rotary_kv_frame` prepares borrowed executor effects.

The offline exporter and tiny independent CUDA probe cover base/YaRN rotary,
both scale policies, all seven non-rotary blocks of a 512-wide head, zero input,
live rows 0/1/3, inactive output preservation and deterministic repetition.
Physical results must be recorded separately; source tests are not GPU proof.

`RotaryCachePrecision` / `rotary_cache_source` handle standalone compact pooled
rows using their explicit group-start positions. They reuse the same frequency
and prefix-simulation lowering as query/KV execution, avoiding both duplicated
numerical policy and redundant query work. Their new prepared connection has
software tests; whole-chain GPU correctness is still pending while the Sparks
are occupied by the real GLM checkpoint diagnostic.
