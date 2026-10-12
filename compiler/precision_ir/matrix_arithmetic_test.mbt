///|
test "matrix affine and sum laws are explicit without changing resource extents" {
  let plan = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=3,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    mix_law=TransposedF32ResidualThenBranch,
    control_matrix_affine_law=FusedMultiplyAdd,
    matrix_sum_law=ButterflyFloat32,
  )
  assert_eq(plan.control_matrix_affine_law(), FusedMultiplyAdd)
  assert_eq(plan.matrix_sum_law(), ButterflyFloat32)
  assert_eq(plan.frame_bytes(), [48L, 192L, 192L, 24576L, 24576L, 24576L])
  try
    @precision_ir.HyperConnectionPrecision::new(
      stream_count=3,
      hidden_width=4096,
      maximum_rows=1,
      sinkhorn_iterations=20,
      control_epsilon=0.000001,
      normalization_epsilon=0.000001,
      mix_law=TransposedF32ResidualThenBranch,
      matrix_sum_law=ButterflyFloat32,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    _ => fail("unexpected illegal reduction error")
  } noraise {
    _ => fail("non-power-of-two butterfly accepted")
  }
}
