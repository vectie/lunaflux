///|
fn unallocated_stage() -> DeepSeekDecoderStage raise {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  DeepSeekDecoderStage::new(
    spec,
    @pipeline.DecoderStageRange::new(
      layer_count=spec.layer_count(),
      first_layer=0,
      end_layer=1,
    ),
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    maximum_text_workspace_bytes=16777216L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
}

///|
test "DeepSeek activation lease preserves stage state on rejected close and resubmit" {
  let stage = unallocated_stage()
  // This exercises lifecycle guards without checkpoint/device allocation. The
  // common activation-link DMA and rank paths have separate device-double tests.
  stage.state = Prepared
  stage.activation_leased = true
  try stage.close() catch {
    @device.DeviceError::Busy(OrderedExecutorClose) => ()
    error => raise error
  } noraise {
    _ => fail("released a leased residual")
  }
  assert_true(stage.state is Prepared)
  try stage.submit(live_rows=1) catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    error => raise error
  } noraise {
    _ => fail("overwrote a leased residual")
  }
  assert_true(stage.state is Prepared)
  stage.activation_leased = false
  for rows in [0, 4] {
    try stage.submit(live_rows=rows) catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue) => ()
      error => raise error
    } noraise {
      _ => fail("accepted invalid rows")
    }
    assert_true(stage.state is Prepared)
  }
  stage.submission_epoch = 0xffffffffffffffffUL
  try stage.submit(live_rows=1) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue) => ()
    error => raise error
  } noraise {
    _ => fail("wrapped submission identity")
  }
  assert_true(stage.state is Prepared)
  stage.state = Fresh
  stage.close()
  stage.close()
}

///|
test "multiple observers retire one DeepSeek submission without re-polling a queue" {
  let stage = unallocated_stage()
  stage.state = Prepared
  stage.active_rows = 2
  stage.submission_epoch = 1UL
  assert_true(stage.poll_submission())
  assert_true(stage.poll_submission())
  stage.state = Fresh
  stage.close()
}
