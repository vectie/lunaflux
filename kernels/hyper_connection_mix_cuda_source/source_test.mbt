///|
test "hyper-connection mix source fixes transpose, round, and alias semantics" {
  let rendered = hyper_connection_mix_bf16_source(
    hidden_width=4096,
    stream_count=4,
    maximum_rows=32,
    function_symbol="mix4",
  )
  let text = @utf8.decode(rendered.source_bytes())
  assert_true(text.contains("output == residual || output == branch"))
  assert_true(
    text.contains("+ (unsigned long long)source) * 4ULL + destination"),
  )
  assert_true(text.contains("__bfloat162float(__float2bfloat16_rn(post"))
  assert_true(text.contains("__bfloat162float(__float2bfloat16_rn(combine"))
  assert_true(text.contains("residual_sum = __fadd_rn"))
  assert_true(
    text.contains("__bfloat162float(__float2bfloat16_rn(residual_sum))"),
  )
  assert_eq(rendered.numerical_contract(), F32ControlsRoundedBf16OrderedMixV1)
}

///|
test "hyper-connection mix source rejects unsafe geometry" {
  try
    hyper_connection_mix_bf16_source(
      hidden_width=4096,
      stream_count=0,
      maximum_rows=32,
      function_symbol="mix4",
    )
  catch {
    InvalidStreamCount => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("zero streams accepted")
  }
}
