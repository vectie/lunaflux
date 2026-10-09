///|
test "timestep weights preserve four F32 checkpoint tensors" {
  let model = @minimax_h3.build(h3_weight_metadata())
  let plan = timestep_weight_packing(model, max_bytes=100000000L)
  assert_eq(plan.copies.length(), 4)
  assert_eq(plan.copies[0].byte_count, 5376L * 256L * 4L)
  assert_eq(plan.copies[2].byte_count, 2688L * 5376L * 4L)
  assert_eq(plan.copies[2].tensor_name, "time_embedder.linear_2.weight")
  assert_eq(plan.second_weight_offset, plan.first_bias_offset + 5376L * 4L)
  assert_eq(plan.byte_count, plan.second_bias_offset + 2688L * 4L)
  ignore(timestep_weight_packing(model, max_bytes=plan.byte_count))
  try timestep_weight_packing(model, max_bytes=plan.byte_count - 1L) catch {
    BudgetExceeded => ()
    _ => fail("wrong timestep weight budget error")
  } noraise {
    _ => fail("short budget accepted")
  }
}
