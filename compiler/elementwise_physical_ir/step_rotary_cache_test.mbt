///|
test "step rotary operands are bounded and independent of layer count" {
  let plan = @elementwise_physical_ir.StepRotaryCachePlan::new(
    maximum_tokens=2048,
    head_dimension=128,
  ).unwrap()
  assert_eq(plan.geometry(), (2048, 64))
  assert_eq(plan.byte_count(), 1048576L)
  assert_eq(plan.cell_offset(2047, 63), Some(1048568L))
  assert_eq(plan.cell_offset(2048, 0), None)
  assert_eq(plan.cell_offset(0, 64), None)
  assert_eq(plan.cell_offset(-1, 0), None)
  assert_eq(
    @elementwise_physical_ir.StepRotaryCachePlan::new(
      maximum_tokens=0,
      head_dimension=128,
    ),
    None,
  )
  assert_eq(
    @elementwise_physical_ir.StepRotaryCachePlan::new(
      maximum_tokens=1,
      head_dimension=127,
    ),
    None,
  )
}
