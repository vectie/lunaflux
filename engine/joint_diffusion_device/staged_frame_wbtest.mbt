///|
test "complete prediction orders timestep input and fifty-layer output work" {
  let stages = prediction_stages(
    [0, 1, 2, 3],
    [4, 5, 6],
    Array::makei(656, i => i + 7),
  )
  assert_eq(stages.length(), 663)
  for i, stage in stages {
    assert_eq(stage, i)
  }
  for (time, input, model) in [([], [1], [2]), ([0], [], [2]), ([0], [1], [])] {
    try prediction_stages(time, input, model) catch {
      @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
      _ => fail("wrong missing-stage error")
    } noraise {
      _ => fail("missing prediction stage accepted")
    }
  }
}
