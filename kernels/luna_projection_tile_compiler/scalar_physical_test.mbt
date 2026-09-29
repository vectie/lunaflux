///|
test "scalar physical folds cover every component exactly once" {
  for extent in [1, 15, 16, 31, 32, 33, 65, 129] {
    for
      reduction in [
        @luna_projection_tile_compiler.OrderedDot,
        StridedPairwiseDot(32),
        StridedPairwiseDot(64),
      ] {
      let plan = @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
        extent~,
        products=2,
        reduction,
      ).unwrap()
      let (_, _, workers) = plan.geometry()
      let seen = Array::make(extent, 0)
      for worker in 0..<workers {
        for iteration in 0..<=extent {
          if plan.read_at(worker~, iteration~) is Some(index) {
            seen[index] += 1
          }
        }
      }
      assert_true(seen.all(count => count == 1))
      assert_eq(plan.product(0), Some((0, 1, 0)))
      assert_eq(plan.product(1), Some((0, 2, 1)))
      assert_eq(plan.product(2), None)
      assert_eq(plan.read_at(worker=-1, iteration=0), None)
      assert_eq(plan.read_at(worker=workers, iteration=0), None)
      assert_eq(plan.read_at(worker=0, iteration=2147483647), None)
    }
  }
}

///|
test "physical scalar tree retains deterministic ordered partial folds" {
  let ordered = @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
    extent=65,
    products=1,
    OrderedDot,
  ).unwrap()
  assert_eq(ordered.combination_distances().length(), 0)
  for workers in [2, 4, 32, 64] {
    let tree = @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
      extent=65,
      products=1,
      StridedPairwiseDot(workers),
    ).unwrap()
    let partials = Array::make(workers, 0)
    for worker in 0..<workers {
      for iteration in 0..<65 {
        if tree.read_at(worker~, iteration~) is Some(index) {
          partials[worker] += index + 1
        }
      }
    }
    for distance in tree.combination_distances() {
      let previous = partials.copy()
      for worker in 0..<workers {
        if tree.combine_from(worker~, distance~) is Some(source) {
          partials[worker] = previous[worker] + previous[source]
        }
      }
    }
    assert_eq(partials[0], 65 * 66 / 2)
    assert_eq(tree.combine_from(worker=0, distance=3), None)
    assert_eq(tree.combine_from(worker=workers - 1, distance=1), None)
  }
  assert_eq(
    @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
      extent=16,
      products=1,
      StridedPairwiseDot(3),
    ),
    None,
  )
  assert_eq(
    @luna_projection_tile_compiler.ProjectionScalarFoldPlan::new(
      extent=16,
      products=0,
      OrderedDot,
    ),
    None,
  )
}

///|
test "operand window models exact ordered tails and safe priming" {
  for fragments in [1, 2, 7, 8, 9, 15, 17, 64] {
    for slots in 1..<=fragments {
      let window = @luna_projection_tile_compiler.ProjectionOperandWindow::new(
        extent=fragments * 16,
        quantum=16,
        slots~,
      ).unwrap()
      let reads : Array[Int] = []
      for iteration in 0..<=((fragments + slots - 1) / slots) {
        for slot in 0..<slots {
          if window.read_at(window=iteration, slot~) is Some(offset) {
            reads.push(offset)
          }
        }
      }
      assert_eq(reads.length(), fragments)
      for index, offset in reads {
        assert_eq(offset, index * 16)
      }
      assert_eq(window.read_at(window=-1, slot=0), None)
      assert_eq(window.read_at(window=0, slot=slots), None)
      assert_eq(window.read_at(window=2147483647, slot=0), None)
    }
  }
  assert_eq(
    @luna_projection_tile_compiler.ProjectionOperandWindow::new(
      extent=16,
      quantum=16,
      slots=2,
    ),
    None,
  )
  assert_eq(
    @luna_projection_tile_compiler.ProjectionOperandWindow::new(
      extent=17,
      quantum=16,
      slots=1,
    ),
    None,
  )
}
