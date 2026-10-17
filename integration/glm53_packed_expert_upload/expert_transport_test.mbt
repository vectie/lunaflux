///|
test "packed transport reaches recurrent and sparse GLM experts and startup budget" {
  let stages = [@matrix.IndexedRows, PackedRows, PackedPipeline].map(transport => {
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
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      expert_transport=transport,
      tensor_group=(0, 2),
    )
  })
  defer (for stage in stages {
    stage.close()
  })
  for i, stage in stages {
    let source = @utf8.decode(stage.source_bytes())
    for layer in [3, 4] {
      assert_eq(source.contains("glm_routed_\{layer}_pack_rows_0"), i != 0)
      assert_eq(source.contains("glm_routed_\{layer}_pack_rows_1"), i != 0)
    }
    assert_eq(source.contains("mbarrier.try_wait.parity"), i == 2)
    assert_eq(source.contains("mbarrier.inval"), i == 2)
  }
}
