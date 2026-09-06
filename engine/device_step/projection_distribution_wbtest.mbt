///|
test "graph bucket geometry follows the selected matrix distribution" {
  for groups in [1, 2, 4, 8, 16] {
    let source = @launch_contract.AotLaunchDimensions::new(
      grid_x=8192,
      grid_y=1,
      grid_z=1,
      block_x=groups * 32,
      block_y=1,
      block_z=1,
      shared_memory_bytes=groups * 1024,
    )
    let dense = ProjectionTensorCoreGridX(source, 1024)
    let up = ProjectionMlpUpGridX(source, 3072)
    let down = ProjectionMlpDownGridX(source, 1024)
    assert_eq(paged_projection_bucket_grid_x(dense, 1), 1024 / groups)
    assert_eq(paged_projection_bucket_grid_x(dense, 8), 64 / groups)
    assert_eq(paged_projection_bucket_grid_x(up, 1), 3072 / groups)
    assert_eq(paged_projection_bucket_grid_x(up, 32), 384 / groups)
    assert_eq(paged_projection_bucket_grid_x(down, 1), 1024 / groups)
    assert_eq(
      paged_projection_bucket_grid_x(down, 32),
      128 / Int::min(4, groups),
    )
    let launch = paged_bucket_launch(dense, 8)
    assert_eq(launch.block_x(), groups * 32)
    assert_eq(launch.shared_memory_bytes(), groups * 1024)
  }
}
