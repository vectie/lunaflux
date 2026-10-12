# Full-head rotary and KV precision simulation

## Explicit complex product rounding

`RotaryProductLaw` is a numerical IR choice shared by KV, standalone tensor
and compact-cache plans. `SeparateFloat32Products` preserves the previous
default source. Opt-in `ComplexFloat32Fma` evaluates real as
`fma(left,cosine,-round(right*sine))` and imaginary as
`fma(right,cosine,round(left*sine))`, then publishes BF16. Inverse rotation
conjugates sine independently of the multiplication law.

One private CUDA renderer implements both laws for all three consumers;
hardware instructions do not enter the model or scheduling layer. The offline
rotary-arithmetic probe reproduces retained production output and distinguishes
frequency/trigonometry differences from product rounding. In its captured
144-row inverse case, the contracted renderer reduces 22 differing BF16 values
to 11; literal original frequencies plus the contracted product match bitwise.
This is not full-model parity, forward/YaRN qualification or production
selection. Ordinary model builders still use the unchanged default.

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
