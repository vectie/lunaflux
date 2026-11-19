# BF16 embedding row insertion

The generic emitter lowers a typed `EmbeddingScatterProgram`. Six parameters
carry step counts, causal positions, a feature map, live feature count, BF16
features and the BF16 residual. No model/tokenizer/encoder dependency enters
this package.

```mbt check
///|
test "prepare bounded word-copy source" {
  let plan = @effects.EmbeddingScatterProgram::new(
    width=2048,
    maximum_rows=32,
    maximum_positions=4096,
    maximum_features=4096,
    owners=128,
  ).unwrap()
  let source = @embedding_scatter_cuda_source.embedding_scatter_source(
    plan,
    symbol="insert_features",
  )
  assert_eq(source.launch(), (32, 1, 128))
  assert_eq(source.parameter_count(), 6)
}
```

`counts[0]==1` selects prefill; `counts[3]` gives live rows. A map value of-1
preserves token embeddings. Mapped rows copy raw16-bit words, preserving BF16
NaNs/sign bits without floating conversion, scaling or accumulation. A zero
feature count disables insertion for startup or later text requests. There is
no shared memory, KV or rotary interpretation.

CPU word/reference and generated-source host probes cover indexing and copying.
Actual CUDA compilation/execution remains target qualification.
