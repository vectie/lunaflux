///|
test "DeepSeek learned text adapter selects official semantics, not GLM mean" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let plan = @deepseek_v4_text_io.learned_text_io_plan(
    spec,
    maximum_rows=1,
    maximum_output_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  assert_eq(plan.stream_count(), spec.hyper_connection_width())
  assert_eq(plan.text().geometry().2, 1)
  assert_eq(plan.control_epsilon(), spec.hyper_connection_epsilon())
  assert_eq(plan.control_normalization_epsilon(), spec.rms_norm_epsilon())
  let owner = @deepseek_v4_text_io.DeepSeekTextIo::new(
    spec,
    maximum_rows=1,
    maximum_output_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  let source = @utf8.decode(owner.source_bytes())
  assert_true(source.contains("deepseek_text_io_stream_reduce"))
  assert_true(source.contains("controls[stream], value"))
  assert_false(source.contains("stream<4"))
  try owner.suffix() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("fresh learned text frame exposed launches")
  }
  owner.close()
  owner.close()
}
