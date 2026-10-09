///|
test "recurrent delta source binds mixed BF16 F32 layouts and serial correctness" {
  let rendered = causal_normalized_recurrent_delta_bf16_f32_source(
    head_count=64,
    key_dimension=128,
    value_dimension=128,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_rows=32,
    maximum_sequences=8,
    function_symbol="test_recurrent_delta",
  )
  assert_eq(rendered.head_count(), 64)
  assert_eq(rendered.key_dimension(), 128)
  assert_eq(rendered.value_dimension(), 128)
  assert_eq(rendered.maximum_rows(), 32)
  assert_eq(rendered.maximum_sequences(), 8)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("blockIdx.x != 0 || threadIdx.x != 0"))
  assert_true(source.contains("const float *log_decay"))
  assert_true(source.contains("const float *initial_state"))
  assert_true(source.contains("float *final_state"))
  assert_true(source.contains("expf(log_decay[qbase + k])"))
  assert_true(source.contains("g < -5.0f || g > 0.0f"))
  assert_true(source.contains("__float2bfloat16_rn(projected)"))
}

///|
test "recurrent delta source rejects invalid geometry epsilon and symbol" {
  for epsilon in [0.0, -0.1, @double.not_a_number] {
    try
      causal_normalized_recurrent_delta_bf16_f32_source(
        head_count=1,
        key_dimension=1,
        value_dimension=1,
        normalization_epsilon=epsilon,
        log_decay_lower_bound=-5.0,
        maximum_rows=1,
        maximum_sequences=1,
        function_symbol="ok",
      )
    catch {
      InvalidNormalizationEpsilon => ()
      _ => fail("wrong epsilon error")
    } noraise {
      _ => fail("invalid epsilon admitted")
    }
  }
  try
    causal_normalized_recurrent_delta_bf16_f32_source(
      head_count=1,
      key_dimension=1,
      value_dimension=1,
      normalization_epsilon=0.000001,
      log_decay_lower_bound=-5.0,
      maximum_rows=1,
      maximum_sequences=1,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol error")
  } noraise {
    _ => fail("invalid symbol admitted")
  }
  try
    causal_normalized_recurrent_delta_bf16_f32_source(
      head_count=1,
      key_dimension=1,
      value_dimension=1,
      normalization_epsilon=0.000001,
      log_decay_lower_bound=0.0,
      maximum_rows=1,
      maximum_sequences=1,
      function_symbol="ok",
    )
  catch {
    InvalidLogDecayLowerBound => ()
    _ => fail("wrong decay-bound error")
  } noraise {
    _ => fail("invalid decay bound admitted")
  }
}
