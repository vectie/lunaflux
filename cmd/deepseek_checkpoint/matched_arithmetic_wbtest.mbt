///|
test "vector RMS replay retains the refined control dot rather than falling back" {
  for
    mode in [
      "replay-layer-refined", "replay-layer-nonlinear", "replay-layer-vector-rms",
    ] {
    assert_eq(matched_control_dot_law(mode), StripedTreeFloat32(partitions=32))
  }
  assert_eq(matched_control_dot_law("replay-layer"), OrderedFloat32)
  assert_eq(
    matched_control_dot_law("replay-layer-hierarchical-controls"),
    GroupedVectorFloat32(
      groups=18,
      partitions=128,
      vector_width=2,
      publication_partitions=4,
      multiply_add=FusedMultiplyAdd,
    ),
  )
  assert_eq(
    matched_control_dot_law("replay-layer-router-arithmetic"),
    matched_control_dot_law("replay-layer-vector-query"),
  )
  assert_eq(
    matched_control_dot_law("replay-layer-shared-block-matrix"),
    matched_control_dot_law("replay-layer-router-arithmetic"),
  )
}
