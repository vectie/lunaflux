///|
test "partitioned maximum keeps numeric storage and partial publication disjoint" {
  let p = @physical_tile_ir.PartitionedMaximum::new(
    partitions=32,
    group_elements=4,
    after_bytes=101L,
  ).unwrap()
  assert_eq(p.partial_offset(), 104L)
  assert_eq(p.allocation_bytes(), 232L)
  assert_eq(p.live_partitions(0), 0)
  assert_eq(p.live_partitions(1), 1)
  assert_eq(p.live_partitions(64), 32)
  for i in 0..<111 {
    assert_eq(p.owner(i.to_int64(), 1), Some(0))
    assert_eq(p.owner(i.to_int64(), 22), Some(i / 4 % 32))
  }
  assert_eq(p.owner(-1L, 4), None)
  assert_eq(p.owner(0L, 0), None)
  assert_true(
    @physical_tile_ir.PartitionedMaximum::new(
      partitions=0,
      group_elements=4,
      after_bytes=1L,
    )
    is None,
  )
  assert_true(
    @physical_tile_ir.PartitionedMaximum::new(
      partitions=257,
      group_elements=4,
      after_bytes=1L,
    )
    is None,
  )
  assert_true(
    @physical_tile_ir.PartitionedMaximum::new(
      partitions=32,
      group_elements=0,
      after_bytes=1L,
    )
    is None,
  )
  assert_true(
    @physical_tile_ir.PartitionedMaximum::new(
      partitions=32,
      group_elements=4,
      after_bytes=9223372036854775807L,
    )
    is None,
  )
}
