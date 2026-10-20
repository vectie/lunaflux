///|
test "synchronous windows preserve scalar decode and explicit reader retirement" {
  for (columns, inner) in [(4096, 4096), (65, 33), (65, 136)] {
    let old = @matrix.MatrixTilePlan::new(
      rows=64,
      columns~,
      inner~,
      law=ShapeAwareFloat32,
    )
    let old_text = @utf8.decode(
      @tiled_matrix_cuda_source.dense_source(old, symbol="probe"),
    )
    for k in [32, 64, 128] {
      let plan = @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        staging=SynchronousTiles(inner_tile=k),
      )
      let text = @utf8.decode(
        @tiled_matrix_cuda_source.dense_source(plan, symbol="probe"),
      )
      assert_true(text.contains("base_k+=\{k}"))
      assert_true(text.contains("kk+=16"))
      assert_false(text.contains("cp.async"))
      assert_eq(
        text.split("__syncthreads();").count(),
        old_text.split("__syncthreads();").count(),
      )
      assert_eq(
        text.split("const int row_base=").to_array()[0],
        old_text.split("const int row_base=").to_array()[0],
      )
      if k == 32 {
        assert_eq(text, old_text)
      }
    }
  }
}

///|
test "export synchronous staging alternatives through actual dense lowering" {
  if @env.get_env_var("LUNA_EXPORT_SYNCHRONOUS_TILES") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN prefill")
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
  ]
  for i, (columns, inner) in shapes {
    for
      j, staging in [
        @matrix.Synchronous,
        SynchronousTiles(inner_tile=64),
        SynchronousTiles(inner_tile=128),
        SynchronousTiles(inner_tile=32),
      ] {
      let plan = @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        staging~,
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
  let functions = [
    for i in 0..<shapes.length() => {
      "{" + [ for j in 0..<4 => "(void*)p\{i}_s\{j}" ].join(",") + "}"
    }
  ]
  println("static void *functions[][4]={\{functions.join(",")}};")
  println(
    "static const int widths[]={\{shapes.map(s=>s.0.to_string()).join(",")}},inners[]={\{shapes.map(s=>s.1.to_string()).join(",")}};",
  )
  println("LUNA_SOURCE_END")
}
