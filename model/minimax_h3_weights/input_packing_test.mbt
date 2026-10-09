///|
test "input projection packing uses hidden-width F32 biases" {
  let model = @minimax_h3.build(h3_weight_metadata())
  let plan = input_weight_packing(model, max_bytes=10000000L)
  assert_eq(plan.copies.length(), 4)
  assert_eq(plan.copies[0].byte_count, 96L * 5376L * 4L)
  assert_eq(plan.copies[1].byte_count, 5376L * 4L)
  assert_eq(plan.copies[2].tensor_name, "audio_proj_in.weight")
  assert_eq(plan.copies[2].byte_count, 32L * 5376L * 4L)
  assert_eq(plan.copies[3].byte_count, 5376L * 4L)
  ignore(input_weight_packing(model, max_bytes=plan.byte_count))
  try input_weight_packing(model, max_bytes=plan.byte_count - 1L) catch {
    BudgetExceeded => ()
    _ => fail("wrong input budget error")
  } noraise {
    _ => fail("short budget accepted")
  }
}
