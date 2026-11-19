# BF16 packed Q/K/V source

`packed_qkv_source` concatenates separate packed BF16 query, key and value rows
into the existing paged-attention input ABI. It copies 16-bit words, preserving
signed zeros and NaN payloads; it introduces no arithmetic, transpose or round.
Launch one 128-thread block per maximum row. The existing `StepCounts` token
count at index three bounds live work. Input planes and output must be disjoint.

```mbt check
///|
test "packed rows have no padding" {
  let plan = @precision.PackedQkvPrecision::new(
    query_width=4096,
    kv_width=512,
    maximum_rows=32,
  )
  assert_eq(plan.output_bytes(), 327680L)
  ignore(@packed_qkv_cuda_source.packed_qkv_source(plan, symbol="pack_qkv"))
}
```

This package emits source. Actual AOT compilation and CUDA execution are separate
qualification steps. The offline word-pattern oracle lives in
`model/bf16_operator_reference`.
