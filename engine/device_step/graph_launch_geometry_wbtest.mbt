///|
test "query-token launch scaling changes only grid x" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=14,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let scaled = paged_scaled_query_token_launch(source, 8)
  assert_eq(
    (
      scaled.grid_x(),
      scaled.grid_y(),
      scaled.grid_z(),
      scaled.block_x(),
      scaled.block_y(),
      scaled.block_z(),
      scaled.shared_memory_bytes(),
    ),
    (8, 14, 1, 128, 1, 1, 0),
  )
}

///|
test "grid-stride query-token launch scales up to its admitted cap" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=16,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  assert_eq(paged_capped_query_token_launch(source, 8).grid_x(), 8)
  assert_eq(paged_capped_query_token_launch(source, 32).grid_x(), 32)
  assert_eq(paged_capped_query_token_launch(source, 256).grid_x(), 32)
}

///|
test "Qwen postprocess V2 caps its grid to the decode bucket" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=256,
    grid_y=32,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  guard paged_qwen_postprocess_bucket_launch(source)
    is QueryTokenCappedGridX(admitted) else {
    fail("Qwen postprocess V2 did not expose a token-capped grid")
  }
  assert_eq(admitted, source)
  assert_eq(paged_bucket_launch(QueryTokenCappedGridX(admitted), 1).grid_x(), 1)
  assert_eq(paged_bucket_launch(QueryTokenCappedGridX(admitted), 8).grid_x(), 8)
}

///|
test "compiler attention buckets launch query tiles instead of token blocks" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=496,
    grid_y=16,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=31952,
  )
  let rule = QueryTileCappedGridX(source, 16)
  assert_eq(paged_bucket_launch(rule, 1).grid_x(), 1)
  assert_eq(paged_bucket_launch(rule, 128).grid_x(), 8)
  assert_eq(paged_bucket_launch(rule, 512).grid_x(), 32)
  assert_eq(paged_bucket_launch(rule, 4096).grid_x(), 256)
}

///|
test "phase-specific compiler decode scales one row per block" {
  let prefill = @launch_contract.AotLaunchDimensions::new(
    grid_x=256,
    grid_y=16,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=31952,
  )
  let decode = @launch_contract.AotLaunchDimensions::new(
    grid_x=32,
    grid_y=8,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=34076,
  )
  let prefill_scaled = paged_bucket_launch(QueryTileCappedGridX(prefill, 16), 8)
  let scaled = paged_bucket_launch(QueryTokenCappedGridX(decode), 8)
  assert_eq(prefill_scaled.grid_x(), 1)
  assert_eq(
    (scaled.grid_x(), scaled.grid_y(), scaled.shared_memory_bytes()),
    (8, 8, 34076),
  )
}

///|
test "projection buckets remove maximum-envelope decode blocks" {
  let dense = @launch_contract.AotLaunchDimensions::new(
    grid_x=4096,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=8192,
  )
  let dense_rule = ProjectionTensorCoreGridX(dense, 2048)
  assert_eq(paged_projection_bucket_grid_x(dense_rule, 1), 256)
  assert_eq(paged_projection_bucket_grid_x(dense_rule, 8), 16)
  assert_eq(paged_projection_bucket_grid_x(dense_rule, 32), 32)
  let scaled = paged_bucket_launch(dense_rule, 8)
  assert_eq(
    (scaled.grid_x(), scaled.block_x(), scaled.shared_memory_bytes()),
    (16, 256, 8192),
  )
}

///|
test "gated MLP buckets independently scale up and down projections" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=3072,
    grid_y=1,
    grid_z=1,
    block_x=512,
    block_y=1,
    block_z=1,
    shared_memory_bytes=16384,
  )
  let up = ProjectionMlpUpGridX(source, 3072)
  let down = ProjectionMlpDownGridX(source, 1024)
  assert_eq(paged_projection_bucket_grid_x(up, 1), 192)
  assert_eq(paged_projection_bucket_grid_x(up, 8), 12)
  assert_eq(paged_projection_bucket_grid_x(up, 32), 24)
  assert_eq(paged_projection_bucket_grid_x(down, 1), 64)
  assert_eq(paged_projection_bucket_grid_x(down, 8), 16)
  assert_eq(paged_projection_bucket_grid_x(down, 32), 32)
}

///|
test "scalar projection bucket keeps only live output work" {
  let source = @launch_contract.AotLaunchDimensions::new(
    grid_x=16384,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let rule = ProjectionScalarGridX(source, 1024)
  assert_eq(paged_projection_bucket_grid_x(rule, 1), 4)
  assert_eq(paged_projection_bucket_grid_x(rule, 8), 32)
  assert_eq(paged_projection_bucket_grid_x(rule, 32), 128)
}
