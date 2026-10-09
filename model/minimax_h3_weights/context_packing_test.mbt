///|
test "context packing preserves BF16 checkpoint and exact budget" {
  let model = @minimax_h3.build(h3_weight_metadata())
  let plan = context_weight_packing(model, max_bytes=60000000L)
  assert_eq(plan.byte_count, 55060992L)
  assert_eq(plan.copies.length(), 2)
  assert_eq(plan.copies[0].tensor_name, "context_embedder.weight")
  assert_eq(plan.copies[0].byte_count, 55050240L)
  assert_eq(plan.copies[1].tensor_name, "context_embedder.bias")
  assert_eq(plan.copies[1].destination_offset, plan.bias_offset)
  assert_eq(plan.copies[1].byte_count, 10752L)
  ignore(context_weight_packing(model, max_bytes=plan.byte_count))
  for budget in [0L, -1L, plan.byte_count - 1L] {
    try context_weight_packing(model, max_bytes=budget) catch {
      BudgetExceeded => ()
      _ => fail("wrong context budget error")
    } noraise {
      _ => fail("short budget accepted")
    }
  }
}
