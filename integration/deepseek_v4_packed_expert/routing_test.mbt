///|
test "DeepSeek early hash and later corrected top-k share numeric routing IR" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  for layer in [0, 2, 3, 42] {
    let plan = @deepseek_v4_packed_expert.routing_compute_plan(
      spec,
      layer_index=layer,
      maximum_rows=2,
      maximum_workspace_bytes=6240L,
      accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    )
    assert_eq(plan.geometry(), (2, 4096, 256, 6))
    assert_eq(plan.weight().representation().format(), @format.Format::Bf16)
    assert_eq(plan.weight().representation().payload_bytes(), 2097152L)
    assert_eq(plan.score(), @precision.MoeRoutingScore::SqrtSoftplus)
    assert_eq(plan.scaling(), 1.5)
    assert_eq(
      plan.selection(),
      if layer < 3 {
        @precision.MoeRoutingSelection::TokenHash(vocabulary_size=129280)
      } else {
        GroupedTopK
      },
    )
    assert_eq(plan.workspace_bytes(), if layer < 3 { 4192L } else { 6240L })
  }
}
