///|
test "packed qkv input views keep output and scalar decay packed independently" {
  let plan = @precision.RecurrentDeltaPrecision::new(
    head_count=32,
    query_key_heads=16,
    key_dimension=128,
    value_dimension=128,
    decay_layout=PerValueHead,
    query_key_row_stride=8192L,
    value_row_stride=8192L,
    maximum_rows=32,
    maximum_sequences=1,
    cache_slots=1,
    normalization_epsilon=0.000001,
    maximum_state_bytes=2097152L,
  )
  let source = @recurrent_delta_execution_cuda_source.recurrent_delta_source(
    plan,
    symbol="packed_delta",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(
    text.contains("qbase=(unsigned long long)row*8192ULL+(head/2)*128ULL"),
  )
  assert_true(
    text.contains("vbase=(unsigned long long)row*8192ULL+head*128ULL"),
  )
  assert_true(
    text.contains("output[((unsigned long long)row*32ULL+head)*128ULL+lane]"),
  )
  assert_true(text.contains("log_decay[(unsigned long long)row*32ULL+head]"))
  assert_eq(plan.query_key_row_stride(), 8192L)
}
