///|
test "ordered reduction places tree levels without reassociation" {
  for lanes in [1, 2, 8, 32, 64, 128, 256, 1024] {
    for subgroup in [1, 2, 8, 32] {
      if subgroup > lanes {
        continue
      }
      let schedule = @luna_tile_ir.LunaTileOrderedReduction::new(
        lanes~,
        subgroup_lanes=subgroup,
      )
      let actual : Array[Int] = []
      for stride in schedule.shared_strides() {
        assert_true(stride >= subgroup)
        actual.push(stride)
      }
      for stride in schedule.subgroup_strides() {
        assert_true(stride < subgroup)
        actual.push(stride)
      }
      let expected : Array[Int] = []
      for stride = lanes / 2; stride > 0; stride = stride / 2 {
        expected.push(stride)
      }
      assert_eq(actual, expected)
    }
  }
  let schedule = @luna_tile_ir.LunaTileOrderedReduction::new(
    lanes=128,
    subgroup_lanes=32,
  )
  assert_eq(schedule.shared_strides(), ReadOnlyArray::from_array([64, 32]))
  assert_eq(
    schedule.subgroup_strides(),
    ReadOnlyArray::from_array([16, 8, 4, 2, 1]),
  )
}

///|
test "ordered reduction rejects unsupported topologies" {
  for geometry in [(0, 1), (3, 1), (128, 3), (32, 64), (2048, 32), (32, 0)] {
    let (lanes, subgroup) = geometry
    try
      @luna_tile_ir.LunaTileOrderedReduction::new(
        lanes~,
        subgroup_lanes=subgroup,
      )
    catch {
      @luna_tile_ir.InvalidProgram(@luna_tile_ir.Reduction) => ()
      _ => fail("unexpected rejection")
    } noraise {
      _ => fail("invalid topology accepted")
    }
  }
}

///|
test "rounded producer forwarding has a bounded register budget" {
  for
    geometry in [
      (1, 128, Some(1)),
      (1024, 128, Some(8)),
      (2048, 128, Some(16)),
      (2049, 128, None),
      (2147483647, 128, None),
    ] {
    let (elements, lanes, expected) = geometry
    assert_eq(
      @luna_tile_ir.luna_tile_retained_row_values(
        elements~,
        lanes~,
        maximum_values_per_lane=16,
      ),
      expected,
    )
  }
  assert_eq(
    @luna_tile_ir.luna_tile_retained_row_values(
      elements=1,
      lanes=0,
      maximum_values_per_lane=16,
    ),
    None,
  )
}

///|
test "subgroup communication preserves the floating point expression tree" {
  for lanes in [32, 64, 128, 256] {
    let schedule = @luna_tile_ir.LunaTileOrderedReduction::new(
      lanes~,
      subgroup_lanes=32,
    )
    // Cancellation makes an illicit subgroup-first reassociation observable.
    let input : Array[Float] = Array::makei(lanes, i => {
      match i % 4 {
        0 => 1.0e20F
        1 => 1.0F
        2 => -1.0e20F
        _ => 3.0F
      }
    })
    let old = input.copy()
    for stride = lanes / 2; stride > 0; stride = stride / 2 {
      for lane in 0..<stride {
        old[lane] = old[lane] + old[lane + stride]
      }
    }
    let shared = input.copy()
    for stride in schedule.shared_strides() {
      for lane in 0..<stride {
        shared[lane] = shared[lane] + shared[lane + stride]
      }
    }
    let registers = Array::makei(32, lane => shared[lane])
    for stride in schedule.subgroup_strides() {
      let prior = registers.copy()
      for lane in 0..<stride {
        registers[lane] = prior[lane] + prior[lane + stride]
      }
    }
    assert_eq(registers[0], old[0])
  }
}
