///|
test "matrix ownership uses supplied subgroup capability without changing arithmetic" {
  let make = streams => {
    @precision.HyperConnectionPrecision::new(
      stream_count=streams,
      hidden_width=4096,
      maximum_rows=64,
      sinkhorn_iterations=20,
      control_epsilon=1.0e-6,
      normalization_epsilon=1.0e-5,
      mix_law=RoundedBf16ControlsAndProducts,
      control_square_law=VectorizedTreeFloat32(partitions=256, vector_width=1),
      control_dot_law=StripedTreeFloat32(partitions=256),
    )
  }
  let four = @hyper_connection_physical_ir.ControlTailFusion::new(
    make(4),
    subgroup_width=32,
  )
  assert_eq(four.geometry(), (64, 256, 16))
  assert_eq(four.matrix_subgroup(), 32)
  assert_eq(four.initial_control_subgroup(), 32)
  assert_eq(
    @hyper_connection_physical_ir.ControlTailFusion::new(
      make(8),
      subgroup_width=64,
    ).initial_control_subgroup(),
    1,
  )
  assert_eq(
    @hyper_connection_physical_ir.ControlTailFusion::new(
      make(4),
      subgroup_width=8,
    ).matrix_subgroup(),
    1,
  )
  assert_eq(
    @hyper_connection_physical_ir.ControlTailFusion::new(
      make(16),
      subgroup_width=32,
    ).matrix_subgroup(),
    1,
  )
  assert_eq(
    @hyper_connection_physical_ir.ControlTailFusion::new(
      make(8),
      subgroup_width=64,
    ).matrix_subgroup(),
    64,
  )
  try
    @hyper_connection_physical_ir.ControlTailFusion::new(
      make(4),
      subgroup_width=3,
    )
  catch {
    @precision.PrecisionPlanError::NoImplementation => ()
    other => raise other
  } noraise {
    _ => fail("invalid capability")
  }
}
