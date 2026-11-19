///|
test "multi-axis sources use exact constants, prefix split halves and BF16 product stages" {
  let plan = @precision.MultiAxisRotaryPrecision::new(
    head_count=16,
    head_width=256,
    rotary_width=64,
    maximum_rows=32,
    maximum_positions=262144,
    axis_for_pair=Array::makei(32, i => i % 3)[:],
    inverse_frequencies=Array::make(32, 1.0F)[:],
  )
  let coefficients = @multi_axis_rotary_cuda_source.coefficients_source(
    plan,
    symbol="coefficients",
  )
  let rotary = @multi_axis_rotary_cuda_source.rotary_source(
    plan,
    symbol="rotary",
  )
  let text = @utf8.decode(rotary.source_bytes()[:])
  assert_eq(coefficients.launch(), (32, 1, 128))
  assert_eq(rotary.launch(), (32, 16, 128))
  assert_eq(coefficients.parameter_count(), 4)
  assert_eq(rotary.parameter_count(), 5)
  assert_eq(plan.coefficient_bytes(), 4096L)
  assert_true(text.contains("right=input[index+32ULL]"))
  assert_true(text.contains("for(int column=64+threadIdx.x"))
  assert_true(text.contains("__bfloat162float(__float2bfloat16_rn(__fmul_rn("))
  let producer = @utf8.decode(coefficients.source_bytes()[:])
  assert_true(producer.contains("1065353216U"))
  assert_true(producer.contains("__float2bfloat16_rn(cosf(angle))"))
  assert_true(producer.contains("__float2bfloat16_rn(sinf(angle))"))
}
