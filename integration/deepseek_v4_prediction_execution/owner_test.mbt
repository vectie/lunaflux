///|
test "prediction execution does not publish unprepared results" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let owner = @deepseek_v4_prediction_execution.DeepSeekPredictionExecution::new(
    spec,
    maximum_main_rows=3,
    maximum_positions=256,
    maximum_workspace_bytes=16777216L,
    maximum_expert_bytes=8589934592L,
    temperature=0.0,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
  )
  try owner.tokens() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorPoll) => ()
    e => raise e
  } noraise {
    _ => fail("unprepared drafts exposed")
  }
  try owner.submit(Predict) catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    e => raise e
  } noraise {
    _ => fail("unprepared predictor enqueued")
  }
  owner.close()
  owner.close()
}
