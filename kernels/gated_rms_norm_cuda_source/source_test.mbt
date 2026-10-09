///|
test "gated RMSNorm renders exact per-head normalize then sigmoid ordering" {
  let rendered = sigmoid_gated_rms_norm_bf16_source(
    head_count=64,
    head_dimension=128,
    maximum_rows=32,
    epsilon=0.00001,
    function_symbol="gated_norm",
  )
  let text = @utf8.decode(rendered.source_bytes())
  assert_eq(rendered.operation(), PerVectorSigmoidGatedRmsNormBf16)
  assert_eq(rendered.head_count(), 64)
  assert_eq(rendered.head_dimension(), 128)
  assert_true(text.contains("((unsigned long long)row * 64ULL"))
  assert_true(text.contains("__fdiv_rn(sum_sq, 128.0f)"))
  assert_true(
    text.contains("__fmul_rn(__bfloat162float(weight[column]), normalized)"),
  )
  assert_true(text.contains("expf(-gate)"))
  assert_true(
    text.contains("__float2bfloat16_rn(__fmul_rn(weighted, sigmoid_gate))"),
  )
  assert_false(text.contains("o_proj"))
}

///|
test "gated RMSNorm rejects invalid geometry epsilon symbol and overflow" {
  try
    sigmoid_gated_rms_norm_bf16_source(
      head_count=0,
      head_dimension=128,
      maximum_rows=1,
      epsilon=0.00001,
      function_symbol="x",
    )
  catch {
    InvalidHeadCount => ()
    _ => fail("wrong head error")
  } noraise {
    _ => fail("zero heads admitted")
  }
  try
    sigmoid_gated_rms_norm_bf16_source(
      head_count=64,
      head_dimension=128,
      maximum_rows=1,
      epsilon=0.0,
      function_symbol="x",
    )
  catch {
    InvalidEpsilon => ()
    _ => fail("wrong epsilon error")
  } noraise {
    _ => fail("zero epsilon admitted")
  }
  try
    sigmoid_gated_rms_norm_bf16_source(
      head_count=64,
      head_dimension=128,
      maximum_rows=1,
      epsilon=0.00001,
      function_symbol="bad-symbol",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol error")
  } noraise {
    _ => fail("bad symbol admitted")
  }
  try
    sigmoid_gated_rms_norm_bf16_source(
      head_count=65536,
      head_dimension=65536,
      maximum_rows=1,
      epsilon=0.00001,
      function_symbol="x",
    )
  catch {
    OutputSizeOverflow => ()
    _ => fail("wrong overflow error")
  } noraise {
    _ => fail("overflow admitted")
  }
}
