///|
test "vector RMS replay retains the refined control dot rather than falling back" {
  for
    mode in [
      "replay-layer-refined", "replay-layer-nonlinear", "replay-layer-vector-rms",
    ] {
    assert_eq(matched_control_dot_law(mode), StripedTreeFloat32(partitions=32))
  }
  assert_eq(matched_control_dot_law("replay-layer"), OrderedFloat32)
}
