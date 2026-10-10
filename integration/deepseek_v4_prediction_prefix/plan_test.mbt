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
