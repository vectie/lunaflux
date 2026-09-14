///|
test "maximum profile bucket retains safe launch specialization instead of artifact envelope" {
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(91),
    max_query_rows=32,
    max_query_tokens=2048,
    max_page_table_entries=9216,
  )
  assert_eq(paged_maximum_profile_query_bound(profile, false, false), None)
  assert_eq(paged_maximum_profile_query_bound(profile, false, true), None)
  assert_eq(paged_maximum_profile_query_bound(profile, true, true), Some(32))
  let bound = paged_maximum_profile_query_bound(profile, true, false).unwrap()
  assert_eq(bound, 2048)
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=158,
    grid_y=16,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=49168,
  )
  let selected = paged_bucket_launch(QueryTileCappedGridX(source, 64), bound)
  assert_eq(selected.grid_x(), 32)
  assert_eq(selected.grid_y(), source.grid_y())
  assert_eq(selected.block_x(), source.block_x())
  assert_eq(selected.shared_memory_bytes(), source.shared_memory_bytes())
  // Grid-stride traversal still covers per-row tail tiles beyond grid size.
  let visited = FixedArray::make(63, 0)
  for block in 0..<selected.grid_x() {
    for tile = block; tile < visited.length(); tile = tile + selected.grid_x() {
      visited[tile] += 1
    }
  }
  for count in visited {
    assert_eq(count, 1)
  }
}
