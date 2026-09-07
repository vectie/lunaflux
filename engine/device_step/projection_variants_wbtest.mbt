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
    { base_function_index: 3, maximum_token_rows: 8, function_index: 9, launch },
    { base_function_index: 2, maximum_token_rows: 8, function_index: 7, launch },
  ]
  for bound in [1, 2, 8] {
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
  assert_true(select_projection_variant(variants, 2, 0) is None)
  assert_true(select_projection_variant(variants, 4, 8) is None)
  let selected = select_projection_variant(variants, 2, 8).unwrap()
  assert_eq(paged_bucket_launch(selected.launch, 8).grid_x(), 16)
  assert_eq(paged_bucket_launch(selected.launch, 1).grid_x(), 32)
}
