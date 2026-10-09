///|
test "DeepSeek selects base versus YaRN per actual layer compression profile" {
  for
    id in [
      "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Pro",
      "deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Flash-0731",
    ] {
    let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(id)
    for layer in 0..<spec.layer_count() {
      let plan = @deepseek_v4_query_kv.rotary_kv_plan(
        spec,
        layer~,
        maximum_rows=1,
        maximum_workspace_bytes=1048576L,
      )
      match (spec.compression_ratio(layer), plan.frequency()) {
        (Some(0), Base(theta~)) => assert_eq(theta, spec.rope_theta())
        (
          Some(4 | 128),
          Yarn(theta~, original_positions~, factor~, beta_fast~, beta_slow~),
        ) => {
          assert_eq(theta, spec.compressed_rope_theta())
          assert_eq(original_positions, spec.original_position_embeddings())
          assert_eq(factor, spec.rope_scaling_factor())
          assert_eq((beta_fast, beta_slow), (32.0, 1.0))
        }
        _ => fail("layer used wrong rotary policy")
      }
      assert_eq(plan.simulation(), (64, 0.0001, PowerOfTwo))
      assert_eq(plan.geometry().2, spec.head_dimension())
    }
    let owner = @deepseek_v4_query_kv.DeepSeekAttentionInputs::new(
      spec,
      layer=0,
      maximum_rows=1,
      maximum_workspace_bytes=1048576L,
    )
    let text = @utf8.decode(owner.source_bytes())
    assert_true(text.contains("deepseek_l0_query_kv_q_b"))
    assert_true(text.contains("deepseek_l0_attention_inputs_simulate"))
    try owner.outputs() catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      error => raise error
    } noraise {
      _ => fail("unprepared attention operands published")
    }
    owner.close()
    owner.close()
  }
}
