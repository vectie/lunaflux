///|
test "parallel HC requires an explicitly declared matching numerical tree" {
  let make = (square, dot) => {
    @precision.HyperConnectionPrecision::new(
      stream_count=4,
      hidden_width=4096,
      maximum_rows=64,
      sinkhorn_iterations=20,
      control_epsilon=1.0e-6,
      normalization_epsilon=1.0e-5,
      mix_law=RoundedBf16ControlsAndProducts,
      control_square_law=square,
      control_dot_law=dot,
    )
  }
  let plan = @hyper_connection_physical_ir.ParallelControlPlan::new(
    make(
      VectorizedTreeFloat32(partitions=256, vector_width=1),
      StripedTreeFloat32(partitions=256),
    ),
  )
  assert_eq(plan.workers(), 256)
  assert_eq(plan.workgroups(1), 1536)
  assert_eq(plan.workspace_bytes(), 6400L)
  for
    (square, dot) in [
      (
        @precision.SquareReductionLaw::OrderedFloat32,
        @precision.DotReductionLaw::OrderedFloat32,
      ),
      (
        VectorizedTreeFloat32(partitions=256, vector_width=1),
        StripedTreeFloat32(partitions=128),
      ),
      (
        VectorizedTreeFloat32(partitions=256, vector_width=4),
        StripedTreeFloat32(partitions=256),
      ),
    ] {
    try
      @hyper_connection_physical_ir.ParallelControlPlan::new(make(square, dot))
    catch {
      @precision.PrecisionPlanError::NoImplementation => ()
      other => raise other
    } noraise {
      _ => fail("undeclared tree accepted")
    }
  }
}
