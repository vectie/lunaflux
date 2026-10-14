///|
test "shape-aware policy reaches actual GLM TP source without changing low-bit arithmetic" {
  let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
    @glm53_spec.Glm53FlashSpec::official(),
    @pipeline.DecoderStageRange::new(layer_count=45, first_layer=0, end_layer=5),
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
    tensor_group=(0, 2),
  )
  defer stage.close()
  let text = @utf8.decode(stage.source_bytes())
  assert_true(text.contains("Existing projection strategy: decode GEMV"))
  assert_true(text.contains("glm_dense_ffn_0_gate_up"))
  assert_true(text.contains("gates[column%4]"))
  assert_true(text.contains("chosen_groups"))
  assert_true(text.contains("k+=32"))
  let routing = @glm53_packed_expert_upload.routing_compute_plan(
    maximum_rows=64,
    maximum_workspace_bytes=1000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
    matrix_law=ShapeAwareFloat32,
  )
  assert_true(
    routing.projection_law() is GroupedVectorFloat32(partitions=32, ..),
  )
}
