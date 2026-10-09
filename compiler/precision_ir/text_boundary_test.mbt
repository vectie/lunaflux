///|
test "ingress needs no output workspace while egress retains its exact ceiling" {
  let plan = @precision_ir.TextIoPrecision::new(
    vocabulary_size=11,
    hidden_width=6,
    stream_count=1,
    maximum_rows=4,
    maximum_output_rows=2,
    normalization_epsilon=1.0e-5,
    maximum_workspace_bytes=0L,
    boundary=Ingress,
  )
  assert_eq(plan.boundary_workspace_bytes(Ingress), 0L)
  assert_eq(plan.boundary_workspace_bytes(Interior), 0L)
  assert_eq(plan.boundary_workspace_bytes(Egress), 112L)
  try
    @precision_ir.TextIoPrecision::new(
      vocabulary_size=11,
      hidden_width=6,
      stream_count=1,
      maximum_rows=4,
      maximum_output_rows=2,
      normalization_epsilon=1.0e-5,
      maximum_workspace_bytes=111L,
      boundary=Egress,
    )
  catch {
    @precision_ir.PrecisionPlanError::MemoryBudgetExceeded => ()
    e => raise e
  } noraise {
    _ => fail("egress ignored its workspace ceiling")
  }
}
