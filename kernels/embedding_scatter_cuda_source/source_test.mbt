///|
test "scatter source preserves BF16 words and skips decode and text rows" {
  let plan = @effects.EmbeddingScatterProgram::new(
    width=2048,
    maximum_rows=32,
    maximum_positions=4096,
    maximum_features=4096,
    owners=128,
  ).unwrap()
  let source = @embedding_scatter_cuda_source.embedding_scatter_source(
    plan,
    symbol="image_scatter",
  )
  assert_eq(source.launch(), (32, 1, 128))
  assert_eq(source.parameter_count(), 6)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("counts[0]!=1"))
  assert_true(text.contains("feature_map[position]"))
  assert_true(text.contains("uint16_t* residual"))
  assert_true(!text.contains("__float2bfloat16"))
  assert_true(
    try
      @embedding_scatter_cuda_source.embedding_scatter_source(
        plan,
        symbol="bad-symbol",
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
