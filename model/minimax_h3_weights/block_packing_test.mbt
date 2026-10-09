///|
test "H3 block packing swaps gate-up and preserves contiguous ABI spans" {
  let model = @minimax_h3.build(h3_weight_metadata())
  let plan = block_weight_packing(model, block=0, max_bytes=2000000000L)
  assert_eq(plan.layer, 0)
  assert_eq(plan.copies.length(), 13)
  assert_eq(plan.copies[11].destination_offset, plan.adaln_offset)
  assert_eq(plan.copies[11].byte_count, 96768L * 2688L * 2L)
  assert_eq(plan.copies[12].destination_offset, plan.adaln_bias_offset)
  assert_eq(plan.copies[6].source_offset, 14336L * 5376L * 2L)
  assert_eq(plan.copies[7].source_offset, 0L)
  assert_eq(plan.copies[6].tensor_name, plan.copies[7].tensor_name)
  assert_eq(plan.copies[6].destination_offset, plan.mlp_offset)
  assert_eq(plan.copies[9].destination_offset, plan.norm_offset)
  for i in 1..<plan.copies.length() {
    assert_eq(
      plan.copies[i].destination_offset,
      plan.copies[i - 1].destination_offset + plan.copies[i - 1].byte_count,
    )
  }
  assert_eq(
    block_weight_packing(model, block=49, max_bytes=plan.byte_count).layer,
    49,
  )
  try
    block_weight_packing(model, block=0, max_bytes=plan.byte_count - 1L)
  catch {
    BudgetExceeded => ()
    _ => fail("wrong block budget error")
  } noraise {
    _ => fail("short weight budget accepted")
  }
}
