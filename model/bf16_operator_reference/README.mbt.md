# BF16 operator references

Offline scalar references for normalization, full-softmax routing, recurrent
controls, packed routed experts and the shared dense MLP/gate. These functions
are numerical fixtures, not a production CPU fallback.
Inputs round to BF16; stage boundaries remain explicit. Norms and router sums
use an ordered FP32 fold. Host `expf`, `log1pf` and square root do not establish
bitwise equivalence with CUDA or the upstream reduction topology.

`Router` prepares scratch once and fills caller-owned packed indices/weights.
Its scratch belongs to one caller at a time. Equal probabilities use ascending
expert IDs; PyTorch does not specify its `topk` tie order. Production comparison
must account for ties and compare ordinary non-tied logits as well.

`Experts` and `DenseMlp` share the staged BF16 SwiGLU reference. Packed experts
consume `[expert,gate-then-up,intermediate,hidden]` and
`[expert,hidden,intermediate]` source layouts. Expert contributions are visited by
ascending ID; the numeric plan selects F32 combination or BF16 after every add.
The latter differs on `256 + 1 - 256`, which is covered by a regression. A composed
test covers routing, routed experts, shared MLP, shared gate and final output.

The multi-axis rotary reference selects T/H/W per frequency and rotates split
halves in a prefix. A fixture distinguishes BF16 product rounding from a lone
final round; the tail remains unchanged. The pointwise gate reference tests
head-interleaved raw Q/gate views and rounds sigmoid before multiplication.

The KV ingress oracle uses raw 16-bit words carried by UInt: concatenation must
preserve signed zero and NaN payloads. Its paged scatter fixture continues over
fragmented pages and checks that prefix entries, unrelated pages and padding
remain unchanged. It is an offline oracle, not a second cache implementation.

```mbt check
///|
test "BF16 reference retains IEEE overflow" {
  assert_eq(@bf16_operator_reference.round_bf16(1.00390625F), 1.0F)
  assert_true(
    @bf16_operator_reference.round_bf16(
      Float::reinterpret_from_uint(0x7f7fffffU),
    ).is_inf(),
  )
}
```
