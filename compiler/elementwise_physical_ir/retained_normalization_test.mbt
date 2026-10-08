///|
test "retained normalization preserves every ordered reduction edge" {
  for owners in [32, 64, 128, 256] {
    let program = @elementwise_physical_ir.NormalizationProgram::new(
      width=1025,
      owners~,
      subgroup_lanes=32,
      residual=true,
      owner_tree=true,
    ).unwrap()
    let retained = program.retained_tree(maximum_values_per_lane=64).unwrap()
    assert_eq(retained.geometry(), (owners, 32, 33))
    // Symbolic, non-associative trees detect changed reduction order exactly.
    let reference = Array::makei(owners, i => i.to_string())
    let lanes = Array::makei(32, lane => {
      Array::makei(owners / 32, item => (lane + item * 32).to_string())
    })
    for stride = owners / 2; stride > 0; stride = stride / 2 {
      for i in 0..<stride {
        reference[i] = "(" + reference[i] + "+" + reference[i + stride] + ")"
      }
    }
    for (destination, source) in retained.local_edges() {
      for lane in lanes {
        lane[destination] = "(" + lane[destination] + "+" + lane[source] + ")"
      }
    }
    for stride in retained.exchange_strides() {
      for lane in 0..<stride {
        lanes[lane][0] = "(" +
          lanes[lane][0] +
          "+" +
          lanes[lane + stride][0] +
          ")"
      }
    }
    assert_eq(lanes[0][0], reference[0])
  }
}

///|
test "retained normalization enforces register and numerical eligibility" {
  let program = @elementwise_physical_ir.NormalizationProgram::new(
    width=2049,
    owners=128,
    subgroup_lanes=32,
    residual=true,
    owner_tree=true,
  ).unwrap()
  assert_true(program.retained_tree(maximum_values_per_lane=64) is None)
  assert_true(program.retained_tree(maximum_values_per_lane=0) is None)
  let subgroup = @elementwise_physical_ir.NormalizationProgram::new(
    width=128,
    owners=128,
    subgroup_lanes=32,
    residual=true,
  ).unwrap()
  assert_true(subgroup.retained_tree(maximum_values_per_lane=64) is None)
}
