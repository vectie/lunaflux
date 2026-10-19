///|
test "strided head matrices own complete products and validate bank extents" {
  let plan = @grouped_matrix_ir.HeadProjectionPlan::new(
    rows=17,
    heads=4,
    weight_heads=2,
    columns=64,
    inner=32,
    head_stride=4096L,
    column_stride=1L,
    inner_stride=64L,
    offset=0L,
    weight_elements=8192L,
  )
  assert_eq(plan.workgroups(), 8)
  assert_eq(plan.heads(), (4, 2))
  assert_eq(plan.strides(), (4096L, 1L, 64L, 0L))
  try
    @grouped_matrix_ir.HeadProjectionPlan::new(
      rows=17,
      heads=4,
      weight_heads=2,
      columns=64,
      inner=32,
      head_stride=4096L,
      column_stride=1L,
      inner_stride=64L,
      offset=0L,
      weight_elements=4096L,
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    _ => fail("wrong error")
  } noraise {
    _ => fail("invalid weight range")
  }
}
