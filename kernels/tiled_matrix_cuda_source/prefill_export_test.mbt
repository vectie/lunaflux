///|
test "prefill pipeline export includes selected dense dimensions and exact arithmetic" {
  if @env.get_env_var("LUNA_EXPORT_PREFILL_PIPELINES") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN prefill")
  let shapes = [(4096, 4096), (4096, 8192), (1536, 4096), (128, 4096)]
  for i, (columns, inner) in shapes {
    for
      j, staging in [
        @matrix.Synchronous,
        Buffered(inner_tile=32),
        Buffered(inner_tile=64),
        Buffered(inner_tile=128),
      ] {
      let p = @matrix.MatrixTilePlan::new(
        rows=64,
        columns~,
        inner~,
        law=ShapeAwareFloat32,
        staging~,
      )
      println(
        @utf8.decode(
          @tiled_matrix_cuda_source.dense_source(
            p,
            symbol="p\{i}_s\{j}",
            partial_float32=true,
          ),
        ),
      )
    }
  }
  println(
    "static void *functions[4][4]={{(void*)p0_s0,(void*)p0_s1,(void*)p0_s2,(void*)p0_s3},{(void*)p1_s0,(void*)p1_s1,(void*)p1_s2,(void*)p1_s3},{(void*)p2_s0,(void*)p2_s1,(void*)p2_s2,(void*)p2_s3},{(void*)p3_s0,(void*)p3_s1,(void*)p3_s2,(void*)p3_s3}};",
  )
  println(
    "static const int widths[4]={4096,4096,1536,128},inners[4]={4096,8192,4096,4096};",
  )
  println("LUNA_SOURCE_END")
}
