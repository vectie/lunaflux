///|
test "shape strategy owns decode and retains matrix and tail coverage" {
  for (inner, columns, rows) in [(256, 512, 17), (33, 65, 17), (4096, 8192, 64)] {
    let plan = @matrix.MatrixTilePlan::new(
      rows~,
      columns~,
      inner~,
      law=ShapeAwareFloat32,
    )
    let source = @utf8.decode(
      @tiled_matrix_cuda_source.dense_source(plan, symbol="shape_projection"),
    )
    assert_true(source.contains("if(live_rows==1)"))
    assert_true(source.contains("mma_sync"))
    assert_true(plan.tile_count() >= (columns + 3) / 4)
    assert_true(
      source.contains(
        if inner % 32 == 0 {
          "__shfl_down_sync"
        } else {
          "inner+=1"
        },
      ),
    )
    if inner == 256 && @env.get_env_var("LUNA_EXPORT_SHAPE_AWARE") == Some("1") {
      println("LUNA_SOURCE_BEGIN")
      println(source)
      println("static constexpr int projection_grid=\{plan.tile_count()};")
      println("LUNA_SOURCE_END")
    }
  }
}
