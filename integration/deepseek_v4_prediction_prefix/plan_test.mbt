///|
test "official DSpark main prefix uses complete ordered captured width" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let plan = @deepseek_v4_prediction_prefix.prediction_main_plan(
    spec,
    maximum_rows=3,
    maximum_workspace_bytes=49152L,
  )
  assert_eq(plan.geometry(), (12288, 4096, 3))
  assert_eq(plan.normalization_epsilon(), spec.rms_norm_epsilon())
  let owner = @deepseek_v4_prediction_prefix.DeepSeekPredictionMain::new(
    spec,
    maximum_rows=3,
    maximum_workspace_bytes=49152L,
  )
  assert_true(owner.source_bytes().length() > 0)
  try owner.output() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared output exposed")
  }
  owner.close()
  owner.close()
  let attention = @deepseek_v4_prediction_prefix.prediction_attention_plan(
    spec,
    maximum_main_rows=256,
    maximum_positions=1048576,
    maximum_state_bytes=458776L,
  )
  assert_eq(attention.geometry(), (64, 512, 128, 256, 5, 1048576))
  let phases = @deepseek_v4_prediction_prefix.prediction_attention_frame(
    spec,
    maximum_main_rows=256,
    maximum_positions=1048576,
    maximum_state_bytes=458776L,
    symbol_prefix="prediction_attention",
  )
  assert_true(phases.source_bytes().length() > 0)
  phases.close()
  for stage in 0..<3 {
    let attention = @deepseek_v4_prediction_prefix.DeepSeekPredictionAttention::new(
      spec,
      stage~,
      maximum_main_rows=8,
      maximum_positions=256,
      maximum_workspace_bytes=4194304L,
    )
    assert_true(attention.workspace_bytes() > 458776L)
    assert_true(attention.source_bytes().length() > 0)
    if stage == 0 &&
      @env.get_env_var("LUNA_EXPORT_PREDICTION_ATTENTION") == Some("1") {
      println("LUNA_ATTENTION_BEGIN")
      println(@utf8.decode(attention.source_bytes()))
      println("LUNA_ATTENTION_END")
    }
    attention.close()
    attention.close()
  }
  try
    @deepseek_v4_prediction_prefix.DeepSeekPredictionAttention::new(
      spec,
      stage=3,
      maximum_main_rows=8,
      maximum_positions=256,
      maximum_workspace_bytes=4194304L,
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("nonexistent prediction stage accepted")
  }
  let base = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-Base",
  )
  try
    @deepseek_v4_prediction_prefix.prediction_main_plan(
      base,
      maximum_rows=1,
      maximum_workspace_bytes=49152L,
    )
  catch {
    @precision.PrecisionPlanError::IllegalPlacement => ()
    e => raise e
  } noraise {
    _ => fail("base profile substituted for DSpark")
  }
}
