///|
test "recurrent tensor plan halves heads and caches with F32 publication" {
  let spec = @glm53_spec.Glm53FlashSpec::official()
  for rank in 0..<2 {
    let plan = @glm53_packed_expert_upload.recurrent_block_plan(
      spec,
      layer_index=0,
      maximum_rows=16,
      maximum_sequences=1,
      cache_slots=1,
      maximum_state_bytes=100000000L,
      matrix_law=MatrixFloat32,
      evaluation=ChunkTriangularFloat32,
      tensor_group=(rank, 2),
    )
    assert_eq(plan.delta().geometry(), (32, 128, 128))
    assert_eq(plan.convolution().geometry(), (4096, 4))
    assert_true(plan.publication() == ReduceFloat32BeforeRound)
  }
}

///|
test "both attention families dense and shared FFN reach tensor collective contract" {
  for rank in 0..<2 {
    let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=45,
        first_layer=0,
        end_layer=45,
      ),
      maximum_rows=16,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=100000000L,
      maximum_layer_workspace_bytes=100000000L,
      maximum_layer_expert_bytes=5000000000L,
      maximum_text_workspace_bytes=10000000L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      recurrent_evaluation=ChunkTriangularFloat32,
      matrix_law=MatrixFloat32,
      tensor_group=(rank, 2),
    )
    defer stage.close()
    assert_eq(stage.text_boundary().has_egress(), rank == 1)
    let contract = stage.tensor_contract(
      @device.TensorCollectiveContractDigest::from_sha256(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      ),
    )
    assert_eq(contract.site_count(), 132)
    let text = @utf8.decode(stage.source_bytes())
    assert_true(text.contains("glm_recurrent_0_16("))
    assert_true(text.contains("glm_sparse_3_output_publish("))
    assert_true(text.contains("glm_dense_ffn_0_publish("))
    assert_true(text.contains("const float *"))
  }
}
