///|
test "Flash DSpark expert compute uses FP4 weights and dynamic block128 FP8 activation" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let plan = @deepseek_v4_packed_expert.expert_compute_plan(
    spec,
    layer_index=0,
    expert_indices=[255, 1],
    maximum_rows=2,
    memory_ceiling=100000000L,
    accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
  )
  assert_eq(plan.geometry(), (2, 4096, 2048, 256, 6))
  assert_eq(plan.activation_rounding(), F32Product)
  assert_eq(plan.routing_placement(), BeforeDownInputRound)
  let (qx, qa) = plan.activation_quantization().unwrap()
  assert_eq(qx.operand().representation().scale_shape(), (2, 32))
  assert_eq(qa.operand().representation().scale_shape(), (12, 16))
  assert_eq(qx.amax_floor(), 0.0001)
  let names = @weights.packed_expert_weights(spec, layer_index=0, expert_indices=[
    255, 1,
  ])
  assert_eq(names[0].payload_name(), "layers.0.ffn.experts.255.w1.weight")
  assert_eq(names[1].scales_name(), "layers.0.ffn.experts.255.w3.scale")
  assert_eq(names[2].representation().shape(), (4096, 2048))
  assert_eq(names[0].representation().scale_shape(), (2048, 128))
}

///|
test "shared expert remains replicated FP8 while routed Flash is FP4" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let plan = @deepseek_v4_packed_expert.shared_expert_compute_plan(
    spec,
    layer_index=0,
    maximum_rows=2,
    memory_ceiling=100000000L,
    accuracy=@precision.Accuracy::new(absolute=0.1, relative=0.1),
  )
  assert_eq(plan.geometry(), (2, 4096, 2048, 1, 1))
  assert_eq(plan.weights()[0].representation().format(), MxFp8E4M3)
  assert_eq(plan.weights()[0].representation().scale_shape(), (16, 32))
  assert_eq(plan.weights()[2].representation().scale_shape(), (32, 16))
  assert_eq(
    plan.activation_quantization().unwrap().1.operand().representation().shape(),
    (2, 2048),
  )
  let names = @weights.packed_shared_expert_weights(spec, layer_index=0)
  assert_eq(names[0].payload_name(), "layers.0.ffn.shared_experts.w1.weight")
  assert_eq(names[2].scales_name(), "layers.0.ffn.shared_experts.w2.scale")
  assert_eq(plan.bank_bytes(), 25167360L)
}
