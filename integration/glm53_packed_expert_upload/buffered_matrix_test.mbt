///|
test "buffered projection schedule reaches actual GLM source without changing bytes owned" {
  let stages = [@matrix.Synchronous, Buffered(inner_tile=32)].map(matrix_staging => {
    @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=45,
        first_layer=3,
        end_layer=5,
      ),
      maximum_rows=64,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=100000000L,
      maximum_layer_workspace_bytes=100000000L,
      maximum_layer_expert_bytes=5000000000L,
      maximum_text_workspace_bytes=10000000L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      matrix_law=ShapeAwareFloat32,
      matrix_staging~,
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      tensor_group=(0, 2),
    )
  })
  defer (for stage in stages {
    stage.close()
  })
  let direct = @utf8.decode(stages[0].source_bytes())
  let buffered = @utf8.decode(stages[1].source_bytes())
  assert_false(direct.contains("physical operand ring=2"))
  assert_true(buffered.contains("physical operand ring=2, K=32"))
  assert_true(buffered.contains("glm_recurrent_4_inputs"))
  assert_true(buffered.contains("storage stride remains 2051"))
}
