///|
test "packed transport reaches recurrent and sparse GLM experts and startup budget" {
  let stages = [
    (@matrix.IndexedRows, 1),
    (PackedRows, 1),
    (PackedPipeline, 1),
    (IndexedBuffered, 1),
    (IndexedRows, 2),
    (IndexedBuffered, 2),
    (PackedBulk(columns=128, consumer_teams=2, workers=48), 2),
    (PackedBulkTiles(columns=128, consumer_teams=2, workers=48), 2),
  ].map(entry => {
    let (transport, slots) = entry
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
      expert_fragment_slots=slots,
      tensor_group=(0, 2),
    )
  })
  defer (for stage in stages {
    stage.close()
  })
  for i, stage in stages {
    let source = @utf8.decode(stage.source_bytes())
    for layer in [3, 4] {
      assert_eq(
        source.contains("glm_routed_\{layer}_pack_rows_0"),
        i is (1 | 2 | 6 | 7),
      )
      assert_eq(
        source.contains("glm_routed_\{layer}_pack_rows_1"),
        i is (1 | 2 | 6 | 7),
      )
    }
    assert_eq(source.contains("mbarrier.try_wait.parity"), i is (2 | 6 | 7))
    assert_eq(source.contains("mbarrier.inval"), i is (2 | 6 | 7))
    assert_eq(source.contains("cp.async.bulk.shared::cta"), i is (6 | 7))
    assert_eq(source.contains("_prepare_bank_copy"), i == 7)
    assert_eq(
      source.contains("bulk rows; producers=32; consumers=256; stages=2"),
      i is (6 | 7),
    )
    assert_eq(source.contains("cooperative operand ring=2"), i is (3 | 5))
    assert_eq(
      source.contains("ordered-operand-reads-v1;items=4;slots=2"),
      i is (4 | 5 | 6 | 7),
    )
  }
}

///|
test "shared column ownership propagates through recurrent and sparse owners" {
  for columns in [16, 32] {
    let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
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
      shared_matrix_columns=columns,
      expert_scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
      tensor_group=(0, 2),
    )
    defer stage.close()
    let source = @utf8.decode(stage.source_bytes())
    for layer in [3, 4] {
      assert_true(source.contains("glm_shared_\{layer}_gate_up"))
    }
    assert_true(
      source.contains("column_base=(blockIdx.x%\{1024/columns})*\{columns}"),
    )
    assert_true(source.contains("float result[\{16*columns}]"))
  }
}
