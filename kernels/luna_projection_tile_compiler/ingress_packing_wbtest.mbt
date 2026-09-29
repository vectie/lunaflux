///|
test "multihead producer changes physical schedule not semantic numerical law" {
  let request = ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
    dispatch=projection_compiler_dispatch(Qkv),
    maximum_rows=32,
    maximum_tokens=256,
    epilogue=ProjectionAttentionIngressEpilogue::new(
      query_heads=4,
      kv_heads=2,
      head_dimension=128,
      norm_epsilon=0.000001,
      rope_theta=1000000.0,
    ),
  )
  let baseline = compile_projection_tile(request)
  for heads in [1, 2, 4] {
    let compiled = compile_projection_tile(
      request.with_ingress_head_tiles(heads),
    )
    assert_eq(compiled.program().digest(), baseline.program().digest())
    assert_eq(
      compiled.schedule().single_row_reduction(),
      baseline.schedule().single_row_reduction(),
    )
    assert_eq(
      compiled.schedule().matrix_pipeline().unwrap().output_tiles(),
      heads * 8,
    )
    let columns = compiled
      .schedule()
      .column_fold_plan(consumer_groups=4, maximum_live_accumulators=2 * heads)
      .unwrap()
    assert_eq(columns.geometry(), (4, 2 * heads, 2 * heads, 1))
    if heads != 1 {
      assert_true(compiled.schedule().digest() != baseline.schedule().digest())
    }
  }
}
