///|
test "head reduction fixes normalized controls layout and one BF16 round" {
  let rendered = try! head_reduce_v1_bf16_f32_source(
    stream_width=4,
    hidden_width=4096,
    maximum_rows=17,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    function_symbol="lf_hc_head",
  )
  assert_eq(rendered.operation(), HeadReduceV1)
  assert_eq(
    rendered.numerical_contract(),
    OrderedF32HeadControlReductionBf16RoundV1,
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("const int32_t live_rows = counts[3]"))
  assert_true(source.contains("live_rows > 17"))
  assert_true(
    source.contains(
      "function_weights[weight_base + (unsigned long long)column]",
    ),
  )
  assert_true(source.contains("const float scalar = scale[0]"))
  assert_true(source.contains("__shared__ float controls[4]"))
  assert_true(source.contains("__fdiv_rn(1.0f, sqrtf(__fadd_rn(mean"))
  assert_true(source.contains("__fadd_rn(dot, __fmul_rn"))
  assert_true(
    source.contains("__fadd_rn(sum, __fmul_rn(controls[stream], value))"),
  )
  assert_eq(source.split("__float2bfloat16_rn").length(), 2)
}

///|
fn head_test_render(
  width : Int,
  hidden : Int,
  rows : Int,
  control : Double,
  normalization : Double,
  symbol : String,
) -> DecoderHyperConnectionHeadReduceCudaSource raise DecoderHyperConnectionCudaSourceError {
  head_reduce_v1_bf16_f32_source(
    stream_width=width,
    hidden_width=hidden,
    maximum_rows=rows,
    control_epsilon=control,
    normalization_epsilon=normalization,
    function_symbol=symbol,
  )
}

///|
test "head reduction rejects unsafe geometry parameters and symbols" {
  try head_test_render(1, 16, 1, 0.001, 0.001, "valid") catch {
    InvalidStreamWidth => ()
    _ => fail("wrong stream-width rejection")
  } noraise {
    _ => fail("unsafe stream width was accepted")
  }
  try head_test_render(4, 0, 1, 0.001, 0.001, "valid") catch {
    InvalidHiddenWidth => ()
    _ => fail("wrong hidden-width rejection")
  } noraise {
    _ => fail("unsafe hidden width was accepted")
  }
  try head_test_render(4, 16, 0, 0.001, 0.001, "valid") catch {
    InvalidMaximumRows => ()
    _ => fail("wrong row-bound rejection")
  } noraise {
    _ => fail("unsafe row bound was accepted")
  }
  try head_test_render(4, 16, 1, 0.0, 0.001, "valid") catch {
    InvalidEpsilon => ()
    _ => fail("wrong epsilon rejection")
  } noraise {
    _ => fail("zero control epsilon was accepted")
  }
  try head_test_render(4, 16, 1, 0.001, 0.001, "bad-symbol") catch {
    InvalidFunctionSymbol => ()
    _ => fail("wrong symbol rejection")
  } noraise {
    _ => fail("unsafe symbol was accepted")
  }
}
