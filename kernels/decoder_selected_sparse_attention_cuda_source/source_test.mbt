///|
test "selected sparse attention source fixes exact shared-KV sink ABI" {
  let rendered = shared_key_value_selected_attention_bf16_source(
    query_head_count=64,
    head_dimension=512,
    selected_width=8320,
    maximum_key_slots=1056768,
    maximum_rows=32,
    maximum_batches=1,
    function_symbol="lunaflux_decoder_selected_sparse_attention_bf16_v1",
  )
  assert_eq(rendered.operation(), SharedKeyValueSelectedAttentionBf16)
  assert_eq(
    rendered.numerical_contract(),
    SerialOrderedF32SinkDenominatorBf16OutputV1,
  )
  assert_eq(rendered.maximum_key_slots(), 1056768)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const int32_t batch_count = counts[2]"))
  assert_true(source.contains("const int32_t live_rows = counts[3]"))
  assert_true(source.contains("float maximum = attention_sink[head]"))
  assert_true(
    source.contains(
      "float denominator = expf(__fsub_rn(attention_sink[head], maximum))",
    ),
  )
  assert_true(source.contains("__fmul_rn"))
  assert_true(source.contains("__fadd_rn"))
  assert_true(source.contains("__fdiv_rn"))
  assert_true(source.contains("__float2bfloat16_rn"))
  assert_false(source.contains("inverse_rope"))
  assert_false(source.contains("topk"))
}

///|
test "selected sparse attention source rejects hostile geometry and symbols" {
  try
    shared_key_value_selected_attention_bf16_source(
      query_head_count=0,
      head_dimension=512,
      selected_width=8320,
      maximum_key_slots=1056768,
      maximum_rows=32,
      maximum_batches=1,
      function_symbol="lunaflux_decoder_selected_sparse_attention_bf16_v1",
    )
  catch {
    InvalidQueryHeadCount => ()
    _ => fail("unexpected invalid-head error")
  } noraise {
    _ => fail("zero heads admitted")
  }
  try
    shared_key_value_selected_attention_bf16_source(
      query_head_count=64,
      head_dimension=512,
      selected_width=8320,
      maximum_key_slots=1056768,
      maximum_rows=32,
      maximum_batches=33,
      function_symbol="lunaflux_decoder_selected_sparse_attention_bf16_v1",
    )
  catch {
    InvalidMaximumBatches => ()
    _ => fail("unexpected invalid-batch error")
  } noraise {
    _ => fail("batch count above rows admitted")
  }
  try
    shared_key_value_selected_attention_bf16_source(
      query_head_count=64,
      head_dimension=512,
      selected_width=8320,
      maximum_key_slots=1056768,
      maximum_rows=32,
      maximum_batches=1,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol error")
  } noraise {
    _ => fail("invalid symbol admitted")
  }
}
