///|
test "selected expert source binds distinct score/index and weight banks" {
  let source = selected_expert_swiglu_bf16_source(
    hidden_size=4096,
    intermediate_size=2048,
    routed_expert_count=288,
    experts_per_token=8,
    activation=BoundedClampedSwiGlu(10.0),
    maximum_rows=32,
    function_symbol="selected_experts",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("if (input == output) return"))
  assert_true(text.contains("const int32_t *selected_indices"))
  assert_true(text.contains("const float *selected_weights"))
  assert_true(text.contains("const __nv_bfloat16 *gate_weights"))
  assert_true(text.contains("fminf(__bfloat162float(gate_raw), 10.0f)"))
  assert_true(text.contains("for (int32_t expert = 0; expert < 288"))
  assert_false(text.contains("shared"))
  assert_false(text.contains("residual"))
}

///|
test "selected expert source rejects invalid capacity and activation" {
  try
    selected_expert_swiglu_bf16_source(
      hidden_size=8,
      intermediate_size=4,
      routed_expert_count=2,
      experts_per_token=3,
      activation=StandardSwiGlu,
      maximum_rows=1,
      function_symbol="x",
    )
  catch {
    InvalidGeometry => ()
    _ => fail("wrong capacity error")
  } noraise {
    _ => fail("over-capacity selection admitted")
  }
}
