///|
test "segmentation preserves every vector and register round exactly once" {
  for counts in [[1, 2, 3], [256, 256, 256], [64, 512, 512], [17, 33, 65]] {
    for workers in [1, 7, 32, 64, 128, 256, 512] {
      let segments = @luna_projection_tile_compiler.partition_projection_transfer_segments(
        counts,
        workers~,
      ).unwrap()
      let total = counts.fold(init=0, (a, b) => a + b)
      let seen = FixedArray::make(total, false)
      for segment in segments {
        let origin = counts[:segment.operand].fold(init=0, (a, b) => a + b)
        for worker in segment.worker_begin..<segment.worker_end {
          let relative = worker + segment.operand_vector_origin
          assert_true(relative >= 0 && relative < counts[segment.operand])
          let vector = origin + relative
          assert_eq(vector, worker + segment.round * workers)
          assert_false(seen[vector])
          seen[vector] = true
        }
      }
      assert_true(seen.all(x => x))
    }
  }
}

///|
test "transfer expansion budget cannot overflow on valid total vector counts" {
  for counts in [[2147483646, 1], [1, 2147483646], [2147483645, 1, 1]] {
    assert_eq(
      @luna_projection_tile_compiler.partition_projection_transfer_segments(
        counts,
        workers=1,
      ),
      None,
    )
  }
  assert_true(
    @luna_projection_tile_compiler.partition_projection_transfer_segments(
      [4094, 1],
      workers=1,
    )
    is Some(_),
  )
  assert_eq(
    @luna_projection_tile_compiler.partition_projection_transfer_segments(
      [4095, 1],
      workers=1,
    ),
    None,
  )
  // Large extents are valid when a compact worker map needs only two segments.
  let compact = @luna_projection_tile_compiler.partition_projection_transfer_segments(
    [2147483646, 1],
    workers=2147483647,
  ).unwrap()
  assert_eq(compact.length(), 2)
  assert_eq(compact[0].worker_end, 2147483646)
  assert_eq(compact[1].worker_begin, 2147483646)
  assert_eq(compact[1].worker_end, 2147483647)
}

///|
test "invalid and overflowing transfer domains are rejected" {
  for counts in [[], [0], [-1], [2147483647, 1]] {
    assert_eq(
      @luna_projection_tile_compiler.partition_projection_transfer_segments(
        counts,
        workers=32,
      ),
      None,
    )
  }
  assert_eq(
    @luna_projection_tile_compiler.partition_projection_transfer_segments(
      [1],
      workers=0,
    ),
    None,
  )
}
