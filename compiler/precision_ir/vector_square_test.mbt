///|
test "vector square fold validates pure ordering without device limits or allocation changes" {
  let selected = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=4096,
    maximum_rows=2,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    mix_law=TransposedF32ResidualThenBranch,
    control_square_law=VectorizedTreeFloat32(partitions=512, vector_width=4),
    control_root_law=ReciprocalSqrtFloat32,
    normalization_square_law=VectorizedTreeFloat32(
      partitions=512,
      vector_width=4,
    ),
  )
  assert_eq(selected.control_root_law(), ReciprocalSqrtFloat32)
  assert_eq(
    selected.normalization_square_law(),
    VectorizedTreeFloat32(partitions=512, vector_width=4),
  )
  assert_eq(selected.frame_bytes().length(), 6)
  for
    law in [
      @precision_ir.SquareReductionLaw::VectorizedTreeFloat32(
        partitions=0,
        vector_width=4,
      ),
      VectorizedTreeFloat32(partitions=7, vector_width=4),
      VectorizedTreeFloat32(partitions=512, vector_width=3),
    ] {
    try
      @precision_ir.HyperConnectionPrecision::new(
        stream_count=4,
        hidden_width=128,
        maximum_rows=1,
        sinkhorn_iterations=2,
        control_epsilon=0.000001,
        normalization_epsilon=0.000001,
        mix_law=TransposedF32ResidualThenBranch,
        control_square_law=law,
      )
    catch {
      IllegalPlacement => ()
      _ => fail("unexpected precision error")
    } noraise {
      _ => fail("invalid fold ordering")
    }
  }
}
