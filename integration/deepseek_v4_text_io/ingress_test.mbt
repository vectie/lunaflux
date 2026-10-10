///|
test "DeepSeek ingress uses replicated embedding without learned head workspace" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let plan = @deepseek_v4_text_io.text_ingress_plan(spec, maximum_rows=64)
  assert_eq(
    plan.geometry(),
    (
      spec.vocabulary_size(),
      spec.hidden_size(),
      spec.hyper_connection_width(),
      64,
      1,
    ),
  )
  assert_eq(plan.boundary_workspace_bytes(Ingress), 0L)
  assert_eq(
    plan.residual_bytes(),
    64L * spec.hidden_size().to_int64() * 4L * 2L,
  )
  let owner = @deepseek_v4_text_io.DeepSeekTextIngress::new(
    spec,
    maximum_rows=64,
  )
  assert_true(
    @utf8.decode(owner.source_bytes()).contains("deepseek_text_ingress_embed"),
  )
  try owner.prefix() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    e => raise e
  } noraise {
    _ => fail("fresh ingress exposed executable effects")
  }
  owner.close()
  owner.close()
}
