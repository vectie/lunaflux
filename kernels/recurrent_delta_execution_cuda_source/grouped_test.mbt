///|
test "grouped qk and scalar decay lower through the existing ordered kernel" {
  let plan = @precision.RecurrentDeltaPrecision::new(
    head_count=32,
    query_key_heads=16,
    key_dimension=128,
    value_dimension=128,
    decay_layout=PerValueHead,
    maximum_rows=8,
    maximum_sequences=1,
    cache_slots=1,
    normalization_epsilon=0.000001,
    maximum_state_bytes=2097152L,
  )
  let source = @recurrent_delta_execution_cuda_source.recurrent_delta_source(
    plan,
    symbol="minicpm_delta",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_eq(source.launch(), (1, 32, 128))
  assert_true(text.contains("row*16ULL+(head/2)"))
  assert_true(text.contains("log_decay[(unsigned long long)row*32ULL+head]"))
  assert_eq(source.workspace_bytes(), 0L)
  assert_true(
    try
      @precision.RecurrentDeltaPrecision::new(
        head_count=32,
        query_key_heads=16,
        key_dimension=128,
        value_dimension=128,
        decay_layout=PerValueHead,
        maximum_rows=8,
        maximum_sequences=1,
        cache_slots=1,
        normalization_epsilon=0.000001,
        maximum_state_bytes=2097152L,
        evaluation=ChunkTriangularFloat32,
      )
    catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
