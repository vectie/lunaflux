///|
test "learned gated pooling separates persistent overlap and compact frame budgets" {
  let plan = @precision_ir.LearnedGatedPoolPrecision::new(
    input_width=4096,
    feature_width=512,
    ratio=4,
    preceding_window_overlap=true,
    maximum_rows=5,
    maximum_positions=1048576,
    norm_epsilon=1.0e-6,
    maximum_state_bytes=65544L,
    maximum_workspace_bytes=45108L,
  )
  assert_eq(plan.state_bytes(), ReadOnlyArray::from_array([32768L, 32768L, 8L]))
  assert_eq(
    plan.frame_bytes(),
    ReadOnlyArray::from_array([20480L, 20480L, 2048L, 2048L, 8L, 20L, 24L]),
  )
  assert_eq(
    plan.parameter_bytes(),
    ReadOnlyArray::from_array([8388608L, 8388608L, 16384L, 1024L]),
  )
  assert_eq(plan.compressed_rows(), 2)
  assert_eq(plan.effect(), @precision_ir.NumericEffect::CacheCommit)
  try
    @precision_ir.LearnedGatedPoolPrecision::new(
      input_width=4096,
      feature_width=512,
      ratio=4,
      preceding_window_overlap=true,
      maximum_rows=5,
      maximum_positions=1048576,
      norm_epsilon=1.0e-6,
      maximum_state_bytes=65543L,
      maximum_workspace_bytes=45108L,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    error => raise error
  } noraise {
    _ => fail("accepted undersized state")
  }
}
