///|
test "base and prediction blocks keep distinct namespaces and absolute routing laws" {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  for stage in 0..<3 {
    let block = @deepseek_v4_spec.DeepSeekV4BlockAddress::PredictionStage(stage)
    assert_eq(block.execution_layer(spec), 43 + stage)
    assert_eq(block.checkpoint_prefix(spec), "mtp.\{stage}.")
    assert_eq(block.symbol_prefix(spec), "deepseek_prediction_s\{stage}")
    assert_false(block.uses_token_hash(spec))
  }
  let base = @deepseek_v4_spec.DeepSeekV4BlockAddress::BaseLayer(0)
  assert_eq(base.checkpoint_prefix(spec), "layers.0.")
  assert_true(base.uses_token_hash(spec))
  let invalid : Array[@deepseek_v4_spec.DeepSeekV4BlockAddress] = [
    BaseLayer(-1),
    BaseLayer(43),
    PredictionStage(-1),
    PredictionStage(3),
  ]
  for bad in invalid {
    try bad.checkpoint_prefix(spec) catch {
      @deepseek_v4_spec.DeepSeekV4SpecError::UnsupportedProfile => ()
    } noraise {
      _ => fail("invalid address widened a base/prediction boundary")
    }
  }
  let base_spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-Base",
  )
  try
    @deepseek_v4_spec.DeepSeekV4BlockAddress::PredictionStage(0).execution_layer(
      base_spec,
    )
  catch {
    @deepseek_v4_spec.DeepSeekV4SpecError::UnsupportedProfile => ()
  } noraise {
    _ => fail("prediction accepted for base-only model")
  }
}
