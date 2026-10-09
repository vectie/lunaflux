///|
test "paired RMSNorm source fixes independent widths and ordered F32 arithmetic" {
  let source = paired_rms_norm_bf16_source(
    first_width=1024,
    second_width=512,
    maximum_rows=32,
    epsilon=0.000001,
    function_symbol="lunaflux_advanced_decoder_query_key_rms_norm_bf16_v1",
  )
  assert_eq(source.operation(), PairedRmsNormBf16)
  assert_eq(source.numerical_contract(), OrderedFloat32ReductionBf16RoundV1)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("live_rows > 32"))
  assert_true(text.contains("stream == 0"))
  assert_true(text.contains("stream == 1"))
  assert_true(text.contains("row * 1024ULL"))
  assert_true(text.contains("row * 512ULL"))
  assert_true(text.contains("__fmul_rn(value, value)"))
  assert_true(text.contains("__fadd_rn"))
  assert_true(text.contains("__fdiv_rn"))
  assert_true(text.contains("__float2bfloat16_rn"))
  let repeated = paired_rms_norm_bf16_source(
    first_width=1024,
    second_width=512,
    maximum_rows=32,
    epsilon=0.000001,
    function_symbol="lunaflux_advanced_decoder_query_key_rms_norm_bf16_v1",
  )
  assert_eq(source.source_bytes(), repeated.source_bytes())
}

///|
test "paired RMSNorm source rejects hostile geometry and symbol" {
  try
    paired_rms_norm_bf16_source(
      first_width=0,
      second_width=512,
      maximum_rows=1,
      epsilon=0.000001,
      function_symbol="paired_norm",
    )
  catch {
    InvalidHiddenSize => ()
    _ => fail("unexpected paired width rejection")
  } noraise {
    _ => fail("zero paired RMSNorm width was admitted")
  }
  try
    paired_rms_norm_bf16_source(
      first_width=1024,
      second_width=512,
      maximum_rows=1048577,
      epsilon=0.000001,
      function_symbol="paired_norm",
    )
  catch {
    InvalidRowCount => ()
    _ => fail("unexpected paired row rejection")
  } noraise {
    _ => fail("oversized paired RMSNorm row envelope was admitted")
  }
  try
    paired_rms_norm_bf16_source(
      first_width=1024,
      second_width=512,
      maximum_rows=1,
      epsilon=0.000001,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected paired symbol rejection")
  } noraise {
    _ => fail("invalid paired RMSNorm symbol was admitted")
  }
}
