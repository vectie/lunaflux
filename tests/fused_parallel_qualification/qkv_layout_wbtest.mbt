///|
test "QKV referee uses standalone column-major asymmetric weights" {
  let input = Array::make(FUSED_QUALIFICATION_INPUT_WIDTH, 0)
  input[0] = fused_qualification_bf16_bits(2.0F)
  input[1] = fused_qualification_bf16_bits(3.0F)
  input[2] = fused_qualification_bf16_bits(5.0F)
  let weights = Array::make(FUSED_QUALIFICATION_INPUT_WIDTH * 2, 0)
  weights[0] = fused_qualification_bf16_bits(1.0F)
  weights[1] = fused_qualification_bf16_bits(2.0F)
  weights[2] = fused_qualification_bf16_bits(3.0F)
  weights[FUSED_QUALIFICATION_INPUT_WIDTH] = fused_qualification_bf16_bits(
    10.0F,
  )
  weights[FUSED_QUALIFICATION_INPUT_WIDTH + 1] = fused_qualification_bf16_bits(
    20.0F,
  )
  weights[FUSED_QUALIFICATION_INPUT_WIDTH + 2] = fused_qualification_bf16_bits(
    30.0F,
  )
  assert_eq(fused_qualification_dot(input, weights, 0, 0), 23.0F)
  assert_eq(fused_qualification_dot(input, weights, 0, 1), 230.0F)
  let old_row_major = fused_qualification_bf16_float(input[0]) *
    fused_qualification_bf16_float(weights[0]) +
    fused_qualification_bf16_float(input[1]) *
    fused_qualification_bf16_float(weights[2]) +
    fused_qualification_bf16_float(input[2]) *
    fused_qualification_bf16_float(weights[4])
  assert_true(old_row_major != 23.0F)
}
