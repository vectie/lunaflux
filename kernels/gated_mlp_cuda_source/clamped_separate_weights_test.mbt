///|
test "separate clamped SwiGLU renderer binds clamp rounds and alias guard" {
  let rendered = separate_gate_up_clamped_swiglu_dense_bf16_source(
    hidden_size=8,
    intermediate_size=16,
    maximum_rows=4,
    clamp_limit=10.0,
    function_symbol="clamped_separate",
  )
  assert_eq(
    rendered.numerical_contract(),
    OrderedF32AccumulationClampedBf16SwiGluDownBf16V1,
  )
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("if (input == output) return"))
  assert_true(source.contains("gate_clamped = fminf(gate_bf16, 10.0f)"))
  assert_true(source.contains("up_clamped = fmaxf(-10.0f"))
  assert_true(source.contains("__float2bfloat16_rn(gate_sum)"))
  assert_true(source.contains("__fmul_rn"))
}
