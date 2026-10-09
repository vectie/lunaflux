///|
test "final block packing preserves BF16 modulation and F32 output heads" {
  let model = @minimax_h3.build(h3_weight_metadata())
  assert_true(model.guidance_distilled())
  let plan = output_weight_packing(model, max_bytes=100000000L)
  assert_eq(plan.copies.length(), 7)
  assert_eq(plan.copies[1].tensor_name, "norm_out.linear.weight")
  assert_eq(plan.copies[1].byte_count, 10752L * 2688L * 2L)
  assert_eq(plan.copies[3].tensor_name, "proj_out.weight")
  assert_eq(plan.copies[3].byte_count, 96L * 5376L * 4L)
  assert_eq(plan.copies[5].byte_count, 32L * 5376L * 4L)
  for i in 1..<plan.copies.length() {
    assert_eq(
      plan.copies[i].destination_offset,
      plan.copies[i - 1].destination_offset + plan.copies[i - 1].byte_count,
    )
  }
  assert_eq(plan.video_offset % 4L, 0L)
  ignore(output_weight_packing(model, max_bytes=plan.byte_count))
  try output_weight_packing(model, max_bytes=plan.byte_count - 1L) catch {
    BudgetExceeded => ()
    _ => fail("wrong output budget error")
  } noraise {
    _ => fail("short budget accepted")
  }
}
