///|
test "export matrix projection actual renderer" {
  let plan = @matrix.MatrixTilePlan::new(
    rows=16,
    columns=8192,
    inner=4096,
    law=MatrixFloat32,
  )
  let source = @tiled_matrix_cuda_source.dense_source(
    plan,
    symbol="glm_recurrent_25_0",
  )
  let text = @utf8.decode(source)
  assert_true(text.contains("mma_sync"))
  assert_eq(plan.tile_count(), 128)
  if @env.get_env_var("LUNA_EXPORT_TILED_MATRIX") == Some("1") {
    println("LUNA_SOURCE_BEGIN tiled-matrix")
    println(text)
    for k in [33, 40] {
      println(
        @utf8.decode(
          @tiled_matrix_cuda_source.dense_source(
            @matrix.MatrixTilePlan::new(
              rows=17,
              columns=65,
              inner=k,
              law=MatrixFloat32,
            ),
            symbol="tail_\{k}",
          ),
        ),
      )
    }
    println("LUNA_SOURCE_END")
  }
}
