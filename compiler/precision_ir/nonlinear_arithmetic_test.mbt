///|
test "hyper controls separate vector rounding from matrix arithmetic without budget changes" {
  let ordinary = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=128,
    maximum_rows=3,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    mix_law=TransposedF32ResidualThenBranch,
  )
  let selected = @precision_ir.HyperConnectionPrecision::new(
    stream_count=4,
    hidden_width=128,
    maximum_rows=3,
    sinkhorn_iterations=20,
    control_epsilon=0.000001,
    normalization_epsilon=0.000001,
    mix_law=TransposedF32ResidualThenBranch,
    control_vector_affine_law=FusedMultiplyAdd,
    control_sigmoid_law=DirectReciprocal,
  )
  assert_eq(ordinary.control_vector_affine_law(), SeparateMultiplyAdd)
  assert_eq(ordinary.control_sigmoid_law(), StableBranches)
  assert_eq(selected.control_vector_affine_law(), FusedMultiplyAdd)
  assert_eq(selected.control_sigmoid_law(), DirectReciprocal)
  assert_eq(ordinary.workspace_bytes(), selected.workspace_bytes())
  assert_eq(ordinary.function_weight_bytes(), selected.function_weight_bytes())
}
