///|
test "stage assembly follows model boundaries rather than slice-local boundaries" {
  for
    (first, end, boundary) in [
      (0, 3, @precision_ir.TextBoundary::Ingress),
      (3, 5, Interior),
      (44, 45, Egress),
      (0, 45, Complete),
    ] {
    let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=45,
        first_layer=first,
        end_layer=end,
      ),
      maximum_rows=1,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=10000000L,
      maximum_layer_workspace_bytes=20000000L,
      maximum_layer_expert_bytes=5000000000L,
      maximum_text_workspace_bytes=627712L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
    )
    defer stage.close()
    assert_eq(stage.layer_range(), (first, end))
    assert_true(stage.text_boundary() == boundary)
    assert_eq(stage.launch_count(), 0)
    try stage.borrow_text_logits() catch {
      @device.DeviceError::InvalidArgument(StreamingTransfer) => ()
      error => raise error
    } noraise {
      _ => fail("fresh stage exposed an unallocated vocabulary output")
    }
    let source = @utf8.decode(stage.source_bytes())
    assert_true(source.contains("glm_text_io_embed("))
    if first == 0 {
      assert_true(source.contains("glm_dense_ffn_0_"))
    }
    // No unprepared stage can be submitted, including a complete layer plan.
    try stage.submit(live_rows=1) catch {
      @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
      e => raise e
    } noraise {
      _ => fail("unprepared model execution accepted")
    }
  }
}

///|
test "stage range cannot silently describe a different model depth" {
  try
    @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=44,
        first_layer=0,
        end_layer=44,
      ),
      maximum_rows=1,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=10000000L,
      maximum_layer_workspace_bytes=20000000L,
      maximum_layer_expert_bytes=5000000000L,
      maximum_text_workspace_bytes=627712L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("wrong complete-model depth accepted")
  }
}

///|
test "explicit chunk numerical policy reaches both dense and MoE recurrent owners" {
  for
    law in [
      @precision_ir.RecurrentEvaluationLaw::OrderedTokenFloat32,
      ChunkTriangularFloat32,
    ] {
    let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
      @glm53_spec.Glm53FlashSpec::official(),
      @pipeline.DecoderStageRange::new(
        layer_count=45,
        first_layer=0,
        end_layer=5,
      ),
      maximum_rows=16,
      maximum_history=64,
      index_layer_norm_epsilon=1.0e-5,
      maximum_layer_state_bytes=100000000L,
      maximum_layer_workspace_bytes=100000000L,
      maximum_layer_expert_bytes=5000000000L,
      maximum_text_workspace_bytes=10000000L,
      accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
      recurrent_evaluation=law,
    )
    defer stage.close()
    let source = @utf8.decode(stage.source_bytes())
    for layer in [0, 4] {
      assert_eq(
        source.contains("glm_recurrent_\{layer}_delta_prepare"),
        law == ChunkTriangularFloat32,
      )
      assert_eq(
        source.contains("glm_recurrent_\{layer}_delta_solve"),
        law == ChunkTriangularFloat32,
      )
      assert_eq(
        source.contains("glm_recurrent_\{layer}_delta_publish"),
        law == ChunkTriangularFloat32,
      )
    }
    assert_true(source.contains("glm_sparse_3_"))
  }
}

///|
test "matrix law reaches recurrent projection and sparse/recurrent expert owners" {
  let stage = @glm53_packed_expert_upload.GlmDecoderStage::new(
    @glm53_spec.Glm53FlashSpec::official(),
    @pipeline.DecoderStageRange::new(layer_count=45, first_layer=0, end_layer=5),
    maximum_rows=16,
    maximum_history=64,
    index_layer_norm_epsilon=1.0e-5,
    maximum_layer_state_bytes=100000000L,
    maximum_layer_workspace_bytes=100000000L,
    maximum_layer_expert_bytes=5000000000L,
    maximum_text_workspace_bytes=10000000L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.125, relative=0.125),
    matrix_law=MatrixFloat32,
  )
  defer stage.close()
  let source = @utf8.decode(stage.source_bytes())
  assert_true(source.contains("mma_sync"))
  for layer in [3, 4] {
    assert_true(source.contains("glm_routed_\{layer}_group("))
    assert_true(source.contains("glm_shared_\{layer}_group("))
  }
  assert_false(source.contains("_delta_solve"))
}
