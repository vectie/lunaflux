///|
test "row variant selection covers the whole captured envelope and preserves fallback" {
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=4096,
  )
  let launch = ProjectionTensorCoreGridX(dimensions, 1024)
  let variants : Array[PagedProjectionVariant] = [
    {
      base_function_index: 2,
      maximum_token_rows: 32,
      function_index: 8,
      launch,
    },
    {
      base_function_index: 3,
      maximum_token_rows: 8,
      function_index: 9,
      launch,
    },
    {
      base_function_index: 2,
      maximum_token_rows: 8,
      function_index: 7,
      launch,
    },
  ]
  for bound in [2, 8] {
    assert_eq(
      select_projection_variant(variants, 2, bound).unwrap().function_index,
      7,
    )
  }
  for bound in [9, 16, 32] {
    assert_eq(
      select_projection_variant(variants, 2, bound).unwrap().function_index,
      8,
    )
  }
  assert_true(select_projection_variant(variants, 2, 33) is None)
  assert_true(select_projection_variant(variants, 2, 1) is None)
  assert_true(select_projection_variant(variants, 2, 0) is None)
  assert_true(select_projection_variant(variants, 4, 8) is None)
  let selected = select_projection_variant(variants, 2, 8).unwrap()
  assert_eq(paged_bucket_launch(selected.launch, 8).grid_x(), 16)
  assert_eq(paged_bucket_launch(selected.launch, 1).grid_x(), 32)
}

///|
test "head residency and paired MLP row variants retain their launch mapping" {
  let head = @launch_contract.AotLaunchDimensions::new(
    grid_x=1187,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=40960,
  )
  let resident = ProjectionTensorCoreGridX(head, 151936)
  assert_eq(paged_bucket_launch(resident, 8).grid_x(), 1187)
  assert_eq(paged_bucket_launch(resident, 8).shared_memory_bytes(), 40960)
  let mlp = @launch_contract.AotLaunchDimensions::new(
    grid_x=96,
    grid_y=1,
    grid_z=1,
    block_x=64,
    block_y=1,
    block_z=1,
    shared_memory_bytes=2048,
  )
  let variants : Array[PagedProjectionVariant] = [
    {
      base_function_index: 0,
      maximum_token_rows: 8,
      function_index: 2,
      launch: ProjectionMlpUpGridX(mlp, 3072),
    },
    {
      base_function_index: 1,
      maximum_token_rows: 8,
      function_index: 3,
      launch: ProjectionMlpDownGridX(mlp, 1024),
    },
  ]
  let up = select_projection_variant(variants, 0, 8).unwrap()
  let down = select_projection_variant(variants, 1, 8).unwrap()
  assert_eq(up.function_index, 2)
  assert_eq(down.function_index, 3)
  assert_eq(paged_bucket_launch(up.launch, 8).grid_x(), 96)
  assert_eq(paged_bucket_launch(down.launch, 8).grid_x(), 32)
  assert_true(select_projection_variant(variants, 0, 9) is None)
  assert_true(select_projection_variant(variants, 1, 9) is None)
  assert_true(select_projection_variant(variants, 0, 1) is None)
}
