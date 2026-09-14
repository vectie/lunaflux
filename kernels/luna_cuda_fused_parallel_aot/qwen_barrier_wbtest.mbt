///|
test "ingress tail tiles keep inactive column owners in block barriers" {
  for dimension in [16, 32, 64, 128] {
    let candidate = qwen_fused_test_candidate(
      head_dimension=dimension,
      maximum_tokens=32,
    )
    let production = lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
      candidate,
      fused_test_limits(),
    )
    let source = @utf8.decode(production.source_bytes())
    let map = @luna_projection_tile_compiler.ProjectionVectorTransferMap::new(
      vectors=dimension / 16,
      owners=4,
    ).unwrap()
    let (_, _, rounds) = map.geometry()
    assert_true(source.contains("#define LF_COLUMN_ROUNDS \{rounds}\n"))
    assert_true(source.contains("column_round < LF_COLUMN_ROUNDS"))
    assert_false(source.contains("column_base < LF_HEAD_DIM; column_base +="))
    assert_true(source.contains("if (partial_row_tile) __syncthreads();"))
    assert_true(source.contains("if (column_base < LF_HEAD_DIM) {"))
    if @env.get_env_var("LUNA_TEST_EXPORT_INGRESS_BARRIERS") == Some("1") {
      println("LUNA_SOURCE_BEGIN ingress-\{dimension}")
      println(source)
      println("LUNA_SOURCE_END")
    }
  }
}

///|
test "unsupported ingress matrix width returns an error before source emission" {
  try qwen_fused_test_candidate(head_dimension=8) catch {
    FusedParallelAotError::FusedParallelUnsupported(StaticResources) => ()
    _ => fail("unexpected unsupported-width error")
  } noraise {
    _ => fail("sub-matrix head width accepted")
  }
}
