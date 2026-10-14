///|
test "shape-aware GLM binds fused inputs and independent concatenated TP2 reductions" {
  for rank in 0..<2 {
    let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=45,
        first_layer=0,
        end_layer=45,
      ),
      maximum_rows=64,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=100000000L,
      maximum_layer_workspace_bytes=100000000L,
      maximum_layer_expert_bytes=5000000000L,
      maximum_text_workspace_bytes=10000000L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      recurrent_evaluation=ChunkTriangularFloat32,
      matrix_law=ShapeAwareFloat32,
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      tensor_group=(rank, 2),
    )
    defer stage.close()
    let contract = stage.tensor_contract(
      @device.TensorCollectiveContractDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
    )
    assert_eq(contract.site_count(), 90)
    let code = @utf8.decode(stage.source_bytes())
    assert_true(code.contains("void glm_recurrent_0_inputs("))
    assert_false(code.contains("void glm_recurrent_0_0("))
    assert_true(code.contains("void glm_recurrent_0_4("))
    assert_true(code.contains("void glm_moe_combine_3_collective_pack("))
    assert_true(code.contains("void glm_moe_combine_3_collective_unpack("))
    assert_true(code.contains("work_tile+=gridDim.x"))
  }
}
