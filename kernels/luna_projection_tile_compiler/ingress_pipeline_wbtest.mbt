///|
test "ingress shares the matrix pipeline choice and effect machinery" {
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
  let base = baseline.schedule().matrix_pipeline().unwrap()
  assert_eq(base.row_width(), 16)
  assert_eq(base.output_tiles(), 8)
  assert_eq(base.tile_count(), 64)
  for stages in [2, 3, 4] {
    for fragments in [1, 2] {
      let compiled = compile_projection_tile(
        request.with_fold_pipeline(
          MatrixFold,
          stages~,
          transfer_tiles=2,
          fragment_stages=fragments,
        ),
      )
      assert_eq(compiled.program().digest(), baseline.program().digest())
      let pipeline = compiled.schedule().matrix_pipeline().unwrap()
      assert_eq(pipeline.stage_count(), stages)
      assert_eq(pipeline.fragment_stage_count(), fragments)
      assert_eq(pipeline.transfer_width(), 32)
      let effects = pipeline.effect_plan(OverlappedTransfer)
      assert_eq(effects.prime_count(), stages - 1)
      assert_eq(effects.pending_group_limit(), stages - 2)
      assert_eq(effects.iteration(), [
        IssueFutureOperand,
        ConsumeCurrentOperand,
        AwaitOperandTransfers,
        PublishOperandSlot,
      ])
    }
  }
}
