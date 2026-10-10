///|
fn stage(
  spec : @spec.DeepSeekV4ModelSpec,
  first : Int,
  end : Int,
  maximum_output_rows? : Int = 1,
) -> @deepseek_v4_decoder.DeepSeekDecoderStage raise {
  @deepseek_v4_decoder.DeepSeekDecoderStage::new(
    spec,
    @pipeline.DecoderStageRange::new(
      layer_count=spec.layer_count(),
      first_layer=first,
      end_layer=end,
    ),
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    maximum_text_workspace_bytes=16777216L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    maximum_output_rows~,
  )
}

///|
test "verification output capacity reaches the actual learned egress source" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let ordinary = stage(spec, spec.layer_count() - 1, spec.layer_count())
  defer ordinary.close()
  let verification = stage(
    spec,
    spec.layer_count() - 1,
    spec.layer_count(),
    maximum_output_rows=3,
  )
  defer verification.close()
  let source = @utf8.decode(verification.source_bytes())
  assert_true(source.contains("if(n <= 0 || n > 3"))
  assert_false(source == @utf8.decode(ordinary.source_bytes()))
  for outputs in [0, 4] {
    try
      stage(
        spec,
        spec.layer_count() - 1,
        spec.layer_count(),
        maximum_output_rows=outputs,
      )
    catch {
      @precision.PrecisionPlanError::IllegalPlacement => ()
      error => raise error
    } noraise {
      _ => fail("output geometry exceeded input rows before preparation")
    }
  }
}

///|
test "DeepSeek stages retain shared layer IR and only their own text boundaries" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let ingress = stage(spec, 0, 2)
  let interior = stage(spec, 2, 4)
  let egress = stage(spec, spec.layer_count() - 1, spec.layer_count())
  defer ingress.close()
  defer interior.close()
  defer egress.close()
  assert_eq(ingress.text_boundary(), Ingress)
  assert_eq(interior.text_boundary(), Interior)
  assert_eq(egress.text_boundary(), Egress)
  let start = @utf8.decode(ingress.source_bytes())
  let middle = @utf8.decode(interior.source_bytes())
  let end = @utf8.decode(egress.source_bytes())
  assert_true(start.contains("deepseek_text_ingress_embed"))
  assert_false(start.contains("deepseek_text_io_stream_reduce"))
  assert_false(middle.contains("deepseek_text_ingress_embed"))
  assert_false(middle.contains("deepseek_text_io_stream_reduce"))
  assert_false(end.contains("deepseek_text_ingress_embed"))
  assert_true(end.contains("deepseek_text_io_stream_reduce"))
  let plain_slice = slice(spec, 2, 4)
  defer plain_slice.close()
  assert_eq(interior.source_bytes(), plain_slice.source_bytes())
  assert_eq(ingress.launch_count(), 0)
  try ingress.input_residual() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("fresh stage exposed an allocation")
  }
  try ingress.submit(live_rows=1) catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    e => raise e
  } noraise {
    _ => fail("unprepared stage executed")
  }
  // Optional automation exports the exact boundary source, not fixture weights.
  if @env.get_env_var("LUNAFLUX_DEEPSEEK_STAGE_SOURCE") is Some("1") {
    println("LUNA_DEEPSEEK_STAGE_BEGIN")
    println(start)
    println(end)
    println("LUNA_DEEPSEEK_STAGE_END")
  }
  ingress.close()
  ingress.close()
}

///|
test "DeepSeek stage refuses a foreign model layer count" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let range = @pipeline.DecoderStageRange::new(
    layer_count=spec.layer_count() + 1,
    first_layer=0,
    end_layer=1,
  )
  try
    @deepseek_v4_decoder.DeepSeekDecoderStage::new(
      spec,
      range,
      maximum_rows=3,
      maximum_context=256,
      maximum_layer_state_bytes=16777216L,
      maximum_layer_workspace_bytes=16777216L,
      maximum_layer_expert_bytes=8589934592L,
      maximum_text_workspace_bytes=16777216L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("foreign stage range accepted")
  }
}
