///|
test "export segmented row ownership through common lowering" {
  if @env.get_env_var("LUNA_EXPORT_FUSED_ROW_REUSE") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN prefill")
  for
    s, (inner, widths) in [
      (4096, [4096, 4096, 4096, 128, 32, 128]),
      (33, [65, 35, 17, 9, 3, 5]),
    ] {
    let grids = []
    for j in 0..<4 {
      let plans = widths.map(columns => {
        @matrix.MatrixTilePlan::new(
          rows=64,
          columns~,
          inner~,
          law=ShapeAwareFloat32,
          tile_rows=if j == 0 || j == 3 { 16 } else { 32 },
          tile_columns=if j == 0 || j == 3 { 64 } else { 32 },
          staging=SynchronousTiles(inner_tile=if j == 2 { 64 } else { 32 }),
        )
      })
      let p = @matrix.SegmentedMatrixPlan::new(plans)
      println(
        @utf8.decode(
          @tiled_matrix_cuda_source.segmented_source(p, symbol="f\{s}_v\{j}"),
        ),
      )
      grids.push(p.tile_count().to_string())
    }
    println(
      "static const int grids\{s}[]={\{grids.join(",")}},widths\{s}[]={\{widths.map(n=>n.to_string()).join(",")}};",
    )
  }
  println(
    "static void *functions[][4]={{(void*)f0_v0,(void*)f0_v1,(void*)f0_v2,(void*)f0_v3},{(void*)f1_v0,(void*)f1_v1,(void*)f1_v2,(void*)f1_v3}};",
  )
  println("LUNA_SOURCE_END")
}
