///|
test "buffered row ownership preserves all four accumulator regions" {
  for k in [32, 64, 128] {
    let plan = @matrix.MatrixTilePlan::new(
      rows=64,
      columns=4096,
      inner=8192,
      law=ShapeAwareFloat32,
      tile_rows=32,
      tile_columns=32,
      staging=Buffered(inner_tile=k),
    )
    assert_eq(plan.tile(), (32, 32, k))
    assert_eq(plan.operand_slots(), 2)
    assert_eq(plan.tile_count(), 1024)
    let source = @utf8.decode(
      @tiled_matrix_cuda_source.dense_source(plan, symbol="buffered_rows"),
    )
    assert_true(source.contains("tile_a+slot*\{32*k}+(warp/2)*16*\{k}+kk"))
    assert_true(source.contains("tile_b+slot*\{32*k}+(warp%2)*16*\{k}+kk"))
    assert_true(source.contains("result+(warp/2)*16*32+(warp%2)*16"))
    assert_true(source.contains("cp.async.wait_group 0"))
    assert_true(source.contains("publish next slot and retire current readers"))
    assert_true(source.contains("if(live_rows==1)"))
  }
}

///|
test "export buffered row reuse through actual lowering" {
  if @env.get_env_var("LUNA_EXPORT_BUFFERED_ROWS") != Some("1") {
    return
  }
  let shapes = [
    (4096, 4096),
    (4096, 8192),
    (8192, 1536),
    (4096, 1536),
    (65, 33),
    (65, 136),
  ]
  println("LUNA_SOURCE_BEGIN prefill")
  for i, (columns, inner) in shapes {
    for j in 0..<4 {
      let plan = @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        tile_rows=32,
        tile_columns=32,
        staging=if j == 0 || j == 3 {
          SynchronousTiles(inner_tile=if inner >= 8192 { 64 } else { 32 })
        } else {
          Buffered(inner_tile=if j == 1 { 64 } else { 128 })
        },
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
