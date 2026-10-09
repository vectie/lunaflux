///|
test "selected output bounds exclude other packed rows" {
  assert_eq(selected_extent(2019, 414, 39744, 5376), 2225664L)
  assert_eq(selected_extent(2433, 37296, 39744, 5376), 37296L * 5376L)
  for (start, count) in [(-1, 2), (3, 0), (39743, 2)] {
    try selected_extent(start, count, 39744, 5376) catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("unexpected range error")
    } noraise {
      _ => fail("accepted invalid media slice")
    }
  }
}

///|
test "output head geometry preserves F32 video patches and stereo audio" {
  let (input, output, bias) = output_projection_bytes(37296L, 5376, 96)
  assert_eq(input, 37296L * 5376L * 4L)
  assert_eq(output, 37296L * 96L * 4L)
  assert_eq(bias, 384L)
  let (_, audio, audio_bias) = output_projection_bytes(414L, 5376, 32)
  assert_eq(audio, 414L * 32L * 4L)
  assert_eq(audio_bias, 128L)
}
