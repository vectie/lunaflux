///|
test "ordered products preserve folds and count all publication storage" {
  let p = @precision.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=16,
    sinkhorn_iterations=20,
    control_epsilon=1.0e-6,
    normalization_epsilon=1.0e-5,
    mix_law=RoundedBf16ControlsAndProducts,
  )
  let plan = OrderedControlPlan::new(p, workers=256, product_tile=256)
  assert_eq(plan.workgroups(0), 16)
  assert_eq(plan.workgroups(1), 384)
  assert_eq(plan.workgroups(2), 16)
  assert_eq(plan.workspace_bytes(), 1600L)
  assert_eq(plan.local_bytes(), 1024L)
  try OrderedControlPlan::new(p, workers=256, product_tile=255) catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    e => raise e
  } noraise {
    _ => fail("partial cooperative tile accepted")
  }
}
