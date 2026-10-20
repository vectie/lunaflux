///|
test "explicit fragment pipeline emits reuse lookahead and epoch waits" {
  let matrix = @matrix.MatrixTilePlan::new(
    rows=64,
    columns=4096,
    inner=8192,
    law=ShapeAwareFloat32,
  )
  let plan = @matrix.MatrixFragmentPipeline::new(matrix, row_tile=32)
  let text = @utf8.decode(
    @tiled_matrix_cuda_source.fragment_pipeline_source(plan, symbol="pipe"),
  )
  for
    instruction in [
      "ldmatrix.sync.aligned.m8n8.x4", "ldmatrix.sync.aligned.m8n8.x2", "mma.sync.aligned.m16n8k16",
      "cp.async.wait_group 1", "retire readers", "if(live_rows==1)", "logical^((logical>>3)&112u)",
    ] {
    assert_true(text.contains(instruction))
  }
  let selected = @matrix.MatrixTilePlan::new(
    rows=64,
    columns=4096,
    inner=8192,
    law=ShapeAwareFloat32,
    staging=FragmentPipeline(row_tile=32),
  )
  assert_eq(
    @tiled_matrix_cuda_source.dense_source(selected, symbol="pipe"),
    @tiled_matrix_cuda_source.fragment_pipeline_source(plan, symbol="pipe"),
  )
}

///|
test "export explicit fragment pipelines" {
  if @env.get_env_var("LUNA_EXPORT_FRAGMENT_PIPELINE") != Some("1") {
    return
  }
  let partial_float32 = @env.get_env_var("LUNA_FRAGMENT_BF16") != Some("1")
  let shapes = [(4096, 4096), (4096, 8192), (1536, 4096), (128, 4096)]
  println("LUNA_SOURCE_BEGIN prefill")
  if !partial_float32 { println("#define LUNA_PROJECTION_BF16_OUTPUT 1") }
  for i, (columns, inner) in shapes {
    let matrix = @matrix.MatrixTilePlan::new(
      rows=64,
      columns~,
      inner~,
      law=ShapeAwareFloat32,
      tile_rows=32,
      tile_columns=32,
      staging=SynchronousTiles(inner_tile=64),
    )
    println(
      @utf8.decode(
        @tiled_matrix_cuda_source.dense_source(
          matrix,
          symbol="p\{i}_s0",
          partial_float32~,
        ),
      ),
    )
    for j in 1..<4 {
      let plan = @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        staging=FragmentPipeline(row_tile=if j == 2 { 64 } else { 32 }),
      )
      println(
        @utf8.decode(
          @tiled_matrix_cuda_source.dense_source(
            plan,
            symbol="p\{i}_s\{j}",
            partial_float32~,
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
