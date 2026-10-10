///|
test "prediction retirement lease excludes base overwrite and close" {
  let stage = unallocated_stage()
  stage.state = Prepared
  stage.prediction_leased = true
  try stage.submit(live_rows=1) catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    e => raise e
  } noraise {
    _ => fail("base overwrote active prediction captures")
  }
  try stage.close() catch {
    @device.DeviceError::Busy(OrderedExecutorClose) => ()
    e => raise e
  } noraise {
    _ => fail("base released the prediction vocabulary")
  }
  assert_true(stage.state is Prepared)
  stage.release_prediction_input()
  try stage.release_prediction_input() catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorPoll) => ()
    e => raise e
  } noraise {
    _ => fail("capture retirement repeated")
  }
  try stage.acquire_prediction_input() catch {
    @device.DeviceError::Busy(OrderedExecutorEnqueue) => ()
    e => raise e
  } noraise {
    _ => fail("uncaptured/unretired base used as predictor input")
  }
  stage.state = Fresh
  stage.close()
}
