///|
test "segmented matrix owns virtual coordinates and retains standalone arithmetic" {
  for
    (inner, widths) in [
      (33, [3, 65, 128, 7, 1, 7]),
      (4096, [4096, 4096, 4096, 128, 32, 128]),
    ] {
    let plans = widths.map(columns => {
      @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
      )
    })
    let plan = @matrix.SegmentedMatrixPlan::new(plans)
    let prefix = if inner == 33 { "tail" } else { "full" }
    let source = @utf8.decode(
      @tiled_matrix_cuda_source.segmented_source(plan, symbol=prefix + "_fused"),
    )
    assert_true(source.contains("tile_id*4+warp"))
    assert_true(source.contains("column+=tile_count*4"))
    assert_true(source.contains("__forceinline__"))
    if @env.get_env_var("LUNA_EXPORT_REGIONS") == Some("1") {
      println("LUNA_SOURCE_BEGIN")
      println("namespace \{if inner==33 { "tail" } else { "full" }} {")
      println(source)
      for i, p in plans {
        println(
          @utf8.decode(
            @tiled_matrix_cuda_source.dense_source(
              p,
              symbol=prefix + "_separate_\{i}",
            ),
          ),
        )
      }
      println(
        "constexpr int K=\{inner}, R=64, widths[6]={\{widths.map(n=>n.to_string()).join(",")}}, grids[6]={\{plans.map(p=>p.tile_count().to_string()).join(",")}}, fused_grid=\{plan.tile_count()};",
      )
      println("}")
      println("LUNA_SOURCE_END")
    }
  }
}
