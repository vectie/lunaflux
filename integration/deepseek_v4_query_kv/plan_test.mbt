///|
test "DeepSeek query prefix includes Query-B and additional BF16 head normalization" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let plan = @deepseek_v4_query_kv.query_kv_plan(
    spec,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  assert_eq(
    plan.geometry(),
    (
      spec.hidden_size(),
      spec.q_lora_rank(),
      spec.attention_head_count(),
      spec.head_dimension(),
      1,
    ),
  )
  let matrices = plan.weight_representations()
  assert_eq(
    matrices[1].shape(),
    (spec.attention_head_count() * spec.head_dimension(), spec.q_lora_rank()),
  )
  let owner = @deepseek_v4_query_kv.DeepSeekQueryKv::new(
    spec,
    layer=0,
    maximum_rows=1,
    maximum_workspace_bytes=1048576L,
  )
  let source = @utf8.decode(owner.source_bytes())
  assert_true(source.contains("deepseek_l0_query_kv_q_b"))
  assert_true(source.contains("deepseek_l0_query_kv_q_head_norm"))
  try owner.outputs() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    error => raise error
  } noraise {
    _ => fail("fresh frame published query outputs")
  }
  owner.close()
  owner.close()
}
