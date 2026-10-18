///|
test "batched GLM stage propagates shared history pages through sparse and recurrent layers" {
  let domain = @history_domain.PagedHistoryDomain::new(
    rows_per_page=16,
    physical_pages=8,
    maximum_context=64,
    maximum_sequences=4,
  ).unwrap()
  let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
    @glm53_spec.Glm53FlashSpec::official(),
    @pipeline.DecoderStageRange::new(layer_count=45, first_layer=3, end_layer=5),
    maximum_rows=64,
    maximum_history=64,
    maximum_sequences=4,
    cache_slots=8,
    maximum_output_rows=4,
    paged_history=domain,
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
  let source = @utf8.decode(stage.source_bytes())
  assert_true(source.contains("page_generations"))
  assert_true(source.contains("row_sequences"))
  assert_true(source.contains("glm_recurrent_4_inputs"))
  assert_false(source.contains("counts[2] != 1"))
  assert_false(source.contains("counts[2]!=1"))
}
