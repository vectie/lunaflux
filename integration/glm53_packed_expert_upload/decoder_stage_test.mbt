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
