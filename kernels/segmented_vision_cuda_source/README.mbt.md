# Segmented BF16 vision primitives

These generic AOT emitters consume explicit numeric geometry and integer
metadata buffers. They import no MiniCPM, robot or HTTP package.

`attention_source` lowers `SegmentedAttentionPrecision`: full, noncausal
attention per supplied row span with BF16 score/probability staging. The first
implementation supports head widths up to128 (including72), segments up to8192
and128 threads per query/head CTA. Shared storage is `4*maximum_segment+512`
bytes; it is static, not additional dynamic launch memory.

```mbt check
///|
test "prepare head72 attention" {
  let plan = @precision.SegmentedAttentionPrecision::new(
    maximum_rows=16384,
    maximum_segment=4096,
    head_count=16,
    head_width=72,
  )
  let source = @segmented_vision_cuda_source.attention_source(
    plan,
    symbol="vision",
  )
  assert_eq(source.launch(), (16384, 16, 128))
  assert_eq(source.shared_bytes(), 16896)
}
```

`counts[3]` gives live rows. Query/key/value storage is row-major
`[row,head,lane]`. Starts/ends are exclusive-end row bounds prepared by the
layout owner; no dense mask, KV cache, rotary or causal interpretation is added.
Normalized BF16 probabilities are retained exactly in F32 shared storage before
the value fold. Scores/probabilities share storage after the denominator pass.

Other emitters unpack NaViT pixels into matrix rows, gather learned positions,
copy raw BF16 row permutations and add the early merger's four-row mean
residual. Their integer indices are admitted once by the layout owner.

Source emission and the CPU source probe are software evidence. Actual NVCC,
CUDA numerical, sanitizer and performance qualification remain pending.
