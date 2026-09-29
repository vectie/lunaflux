///|
test "operand storage gives each producer vector one disjoint consumer region" {
  for columns in [16, 32, 64, 128] {
    for rotations in [1, 2] {
      let layout = @physical_tile_ir.OperandLayout::new(
        columns~,
        vector=8,
        rotations~,
        row_group=3,
      ).unwrap()
      for owners in [1, 7, 32, 256] {
        let storage = @physical_tile_ir.OperandStorage::new(
          layout,
          [3, 16, 7],
          owners~,
          stages=3,
        ).unwrap()
        assert_eq(storage.slot_elements(), 26 * columns)
        assert_eq(storage.allocation_elements(), 78 * columns)
        assert_true(storage.region(-1) is None)
        assert_true(storage.region(3) is None)
        let seen = FixedArray::make(storage.slot_elements(), false)
        for operand in 0..<3 {
          let region = storage.region(operand).unwrap()
          let (base, rows, width) = region.geometry()
          assert_eq(width, columns)
          let (_, _, rounds) = region.ownership().geometry()
          for owner in 0..<owners {
            for round in 0..<rounds {
              if region.copy_at(owner, round) is Some((row, column, offset)) {
                assert_true(
                  offset >= base && offset + 8 <= base + rows * columns,
                )
                assert_eq(offset % 8, 0)
                for element in 0..<8 {
                  // The independently addressed consumer sees the same value.
                  assert_eq(
                    region.offset(row, column + element),
                    Some(offset + element),
                  )
                  assert_false(seen[offset + element])
                  seen[offset + element] = true
                }
              }
            }
            assert_true(region.copy_at(owner, rounds) is None)
          }
          assert_true(region.offset(rows, 0) is None)
          assert_true(region.offset(-1, 0) is None)
          assert_true(region.offset(0, columns) is None)
        }
        for present in seen {
          assert_true(present)
        }
      }
    }
  }
}

///|
test "storage planning rejects invalid stage sizes before allocation" {
  let scalar = @physical_tile_ir.OperandLayout::new(
    columns=1,
    vector=1,
    rotations=1,
    row_group=1,
  ).unwrap()
  for rows in [[], [0], [-1], [2147483647, 1]] {
    assert_true(
      @physical_tile_ir.OperandStorage::new(scalar, rows, owners=1, stages=1)
      is None,
    )
  }
  assert_true(
    @physical_tile_ir.OperandStorage::new(scalar, [1], owners=0, stages=1)
    is None,
  )
  assert_true(
    @physical_tile_ir.OperandStorage::new(scalar, [1], owners=1, stages=0)
    is None,
  )
  assert_true(
    @physical_tile_ir.OperandStorage::new(
      scalar,
      [1073741824],
      owners=1,
      stages=2,
    )
    is None,
  )
  let maximum = @physical_tile_ir.OperandStorage::new(
    scalar,
    [2147483647],
    owners=256,
    stages=1,
  ).unwrap()
  assert_eq(maximum.allocation_elements(), 2147483647)
  assert_eq(maximum.region(0).unwrap().offset(2147483646, 0), Some(2147483646))
}

///|
test "storage identity binds layout owner stage and ordered region choices" {
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=64,
    vector=8,
    rotations=8,
    row_group=1,
  ).unwrap()
  let baseline = @physical_tile_ir.OperandStorage::new(
      layout,
      [16, 32],
      owners=128,
      stages=2,
    )
    .unwrap()
    .canonical()
  assert_eq(
    baseline,
    @physical_tile_ir.OperandStorage::new(
      layout,
      [16, 32],
      owners=128,
      stages=2,
    )
    .unwrap()
    .canonical(),
  )
  for
    (rows, owners, stages) in [
      ([32, 16], 128, 2),
      ([16, 32], 64, 2),
      ([16, 32], 128, 3),
    ] {
    assert_true(
      baseline !=
      @physical_tile_ir.OperandStorage::new(layout, rows, owners~, stages~)
      .unwrap()
      .canonical(),
    )
  }
}

///|
test "vector ownership indexing is total at integer boundaries" {
  assert_true(
    @physical_tile_ir.VectorOwnership::new(vectors=0, owners=1) is None,
  )
  assert_true(
    @physical_tile_ir.VectorOwnership::new(vectors=1, owners=0) is None,
  )
  let map = @physical_tile_ir.VectorOwnership::new(
    vectors=2147483647,
    owners=256,
  ).unwrap()
  let (_, _, rounds) = map.geometry()
  assert_eq(map.vector_at(254, rounds - 1), Some(2147483646))
  assert_true(map.vector_at(255, rounds - 1) is None)
  assert_true(map.vector_at(0, 2147483647) is None)
  assert_true(map.vector_at(-1, 0) is None)
  assert_true(map.vector_at(256, 0) is None)
  assert_true(map.vector_at(0, -1) is None)
}

///|
test "flattening strips is allowed only when it preserves row permutations" {
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=64,
    vector=8,
    rotations=8,
    row_group=1,
  ).unwrap()
  let aligned = @physical_tile_ir.OperandStorage::new(
    layout,
    [16, 32, 32],
    owners=128,
    stages=2,
  ).unwrap()
  assert_true(aligned.has_concatenated_layout())
  let unaligned = @physical_tile_ir.OperandStorage::new(
    layout,
    [3, 32, 32],
    owners=128,
    stages=2,
  ).unwrap()
  assert_false(unaligned.has_concatenated_layout())
}
