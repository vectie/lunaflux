///|
test "row reuse is independent physical ownership and preserves decode" {
  for k in [32, 64] {
    let plan = @matrix.MatrixTilePlan::new(
      rows=64,
      columns=4096,
      inner=4096,
      law=ShapeAwareFloat32,
      tile_rows=32,
      tile_columns=32,
      staging=SynchronousTiles(inner_tile=k),
    )
    assert_eq(plan.tile(), (32, 32, k))
    assert_eq(plan.tile_count(), 1024)
    let text = @utf8.decode(
      @tiled_matrix_cuda_source.dense_source(plan, symbol="reuse"),
    )
    assert_true(text.contains("(warp/2)*16*\{k}+kk"))
    assert_true(text.contains("(warp%2)*16*\{k}+kk"))
    assert_true(text.contains("if(live_rows==1)"))
    assert_true(text.contains("kk+=16"))
  }
}

///|
test "export row reuse alternatives through actual lowering" {
  if @env.get_env_var("LUNA_EXPORT_ROW_REUSE") != Some("1") {
    return
  }
  let shapes = [
    (4096, 4096),
    (4096, 8192),
    (1536, 4096),
    (128, 4096),
    (512, 4096),
    (16384, 512),
    (4096, 1536),
    (32, 4096),
    (8192, 1536),
    (65, 33),
    (65, 136),
    (4096, 128),
  ]
  println("LUNA_SOURCE_BEGIN prefill")
  for i, (columns, inner) in shapes {
    for j in 0..<4 {
      let plan = @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        tile_rows=if j == 0 || j == 3 { 16 } else { 32 },
        tile_columns=if j == 0 || j == 3 { 64 } else { 32 },
        staging=SynchronousTiles(inner_tile=if j == 2 { 64 } else { 32 }),
      )
      println(
        @utf8.decode(
          @tiled_matrix_cuda_source.dense_source(
            plan,
            symbol="p\{i}_s\{j}",
            partial_float32=true,
          ),
        ),
      )
    }
  }
  println(
    "static void *functions[][4]={\{[for i in 0..<shapes.length() => "{"+[for j in 0..<4 => "(void*)p\{i}_s\{j}"].join(",")+"}"].join(",")}};",
  )
  println(
    "static const int widths[]={\{shapes.map(s=>s.0.to_string()).join(",")}},inners[]={\{shapes.map(s=>s.1.to_string()).join(",")}};",
  )
  println("LUNA_SOURCE_END")
}
