///|
test "single-token prefill selects its own owner before multi-token variants" {
  let (table, profile, dispatch) = graph_dispatch_fixture(7)
  let slots_before = table.bucket_count()
  for bound in paged_prefill_capture_bounds(profile) {
    // Use each owner's token capacity as its test-only identity. Production
    // stores executor indices in the same fixed, startup-populated slot table.
    map_paged_prefill_capture_owner(
      table,
      profile.max_query_rows(),
      bound,
      bound,
      dispatch.slot_owner_indices,
    )
  }
  let one = graph_dispatch_bucket(table, PrefillGraph, 1, 1, 1)
  let selected_bound = dispatch.select(one)
  assert_eq(selected_bound, 1)
  for tokens in [2, 7, 8] {
    assert_eq(
      dispatch.select(
        graph_dispatch_bucket(table, PrefillGraph, 1, tokens, 1024),
      ),
      8,
    )
  }
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, MixedGraph, 2, 2, 1024)),
    8,
  )
  assert_eq(
    dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 1024)),
    7,
  )
  assert_eq(table.bucket_count(), slots_before)
  assert_eq(dispatch.slot_owner_indices.length(), slots_before)
}

///|
test "single-token prefill covers every scalar projection output column" {
  let (table, profile, dispatch) = graph_dispatch_fixture(7)
  for bound in paged_prefill_capture_bounds(profile) {
    map_paged_prefill_capture_owner(
      table,
      profile.max_query_rows(),
      bound,
      bound,
      dispatch.slot_owner_indices,
    )
  }
  let bound = dispatch.select(
    graph_dispatch_bucket(table, PrefillGraph, 1, 1, 1),
  )
  let base = @launch_contract.AotLaunchDimensions::new(
    grid_x=2048,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=8192,
  )
  let small = @launch_contract.AotLaunchDimensions::new(
    grid_x=4096,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=1024,
  )
  let variants : Array[PagedProjectionVariant] = [
    {
      base_function_index: 0,
      maximum_token_rows: 8,
      function_index: 1,
      launch: ProjectionTensorCoreGridX(small, 4096),
    },
  ]
  let variant = select_projection_variant(variants, 0, bound)
  let rule = match variant {
    Some(value) => value.launch
    None => ProjectionTensorCoreGridX(base, 4096)
  }
  let launch = paged_bucket_launch(rule, bound)
  // The single-token branch assigns one column per subgroup, without a
  // grid-stride loop. The old bound=8 plan wrote only 256 of 4096 columns.
  assert_eq(launch.grid_x() * paged_projection_workgroups(launch), 4096)
  assert_true(variant is None)
  assert_eq(launch.grid_x(), 512)
  assert_eq(launch.block_x(), 256)
}

///|
test "single-token prefill preserves independent projection and MLP companion grids" {
  let (table, profile, dispatch) = graph_dispatch_fixture(7)
  for bound in paged_prefill_capture_bounds(profile) {
    map_paged_prefill_capture_owner(
      table,
      profile.max_query_rows(),
      bound,
      bound,
      dispatch.slot_owner_indices,
    )
  }
  let bound = dispatch.select(
    graph_dispatch_bucket(table, PrefillGraph, 1, 1, 1),
  )
  let dimensions = fn(grid : Int, threads : Int, shared : Int) raise {
    @launch_contract.AotLaunchDimensions::new(
      grid_x=grid,
      grid_y=1,
      grid_z=1,
      block_x=threads,
      block_y=1,
      block_z=1,
      shared_memory_bytes=shared,
    )
  }
  let output = dimensions(512, 256, 8192)
  let up = dimensions(768, 512, 16384)
  let down = dimensions(1024, 128, 4096)
  for
    entry in [
      (ProjectionTensorCoreGridX(output, 1024), 128, 256, 8192),
      (ProjectionMlpUpGridX(up, 3072), 192, 512, 16384),
      (ProjectionMlpDownGridX(down, 1024), 256, 128, 4096),
    ] {
    let (rule, expected_grid, expected_threads, expected_shared) = entry
    let launch = paged_bucket_launch(rule, bound)
    assert_eq(launch.grid_x(), expected_grid)
    assert_eq(launch.block_x(), expected_threads)
    assert_eq(launch.shared_memory_bytes(), expected_shared)
    assert_eq(launch.grid_y(), 1)
    assert_eq(launch.grid_z(), 1)
  }
}

///|
test "single-token prefill owns eligible split and wide routes without changing decode" {
  for wide in [false, true] {
    let (table, profile, base) = graph_dispatch_fixture(7)
    let dispatch : PagedOrderedExecutorDispatch = if wide {
      {
        ..base,
        prefill_wide_owner_index: 17,
        prefill_wide_minimum_query_tokens: 1,
      }
    } else {
      {
        ..base,
        prefill_split_owner_index: 17,
        prefill_split_minimum_context_tokens: 512,
        prefill_split_query_tile: 32,
        prefill_split_partition_count: 2,
      }
    }
    let owners = if wide {
      dispatch.prefill_wide_slot_owner_indices
    } else {
      dispatch.prefill_split_slot_owner_indices
    }
    for bound in paged_prefill_capture_bounds(profile) {
      map_paged_prefill_capture_owner(table, 1, bound, bound, owners)
    }
    assert_eq(
      dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 1, 1024)),
      1,
    )
    assert_eq(
      dispatch.select(graph_dispatch_bucket(table, PrefillGraph, 1, 7, 1024)),
      8,
    )
    assert_eq(
      dispatch.select(graph_dispatch_bucket(table, DecodeGraph, 1, 1, 1024)),
      7,
    )
    assert_eq(owners.length(), table.bucket_count())
  }
}

///|
test "single-token prefill adds one bounded startup owner and never duplicates maximum" {
  for
    entry in [
      (1, []),
      (2, [1]),
      (8, [1]),
      (64, [1, 8, 16, 32]),
      (1024, [1, 8, 16, 32, 64, 128, 256, 512]),
      (4096, [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
    ] {
    let (maximum, expected) = entry
    let profile = @launch_contract.PagedKernelProfile::new(
      id=@launch_contract.KernelProfileId::from_stable_int(93),
      max_query_rows=Int::min(32, maximum),
      max_query_tokens=maximum,
      max_page_table_entries=256,
    )
    let bounds = paged_prefill_capture_bounds(profile)
    assert_eq(bounds, ReadOnlyArray::from_array(expected))
    for index, bound in bounds {
      assert_true(bound > 0 && bound < maximum)
      if index > 0 {
        assert_true(bounds[index - 1] < bound)
      }
    }
  }
}
