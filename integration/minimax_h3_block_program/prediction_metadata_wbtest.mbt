///|
test "prediction metadata shares indices but never aliases distinct step values" {
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=4,
    text_rows=1,
    video_rows=2,
    audio_rows=2,
    conditioning=[Video(audio_rows=1, video_rows=1)],
    max_rows=64,
  )
  let schedule = sequence.schedule(
    video=[0.2, 0.25, 0.75],
    audio=[0.1, 0.15, 0.75],
    visual_noise=0.5,
    audio_noise=0.3,
    vision_text_rows=[0],
    max_steps=3,
    max_index_bytes=1024L,
  )
  let layout = prediction_metadata_layout(schedule, 1060L)
  assert_eq(layout.bytes, 1060L)
  inspect(layout.times[:].to_owned(), content="[0, 16, 32]")
  inspect(layout.counts[:].to_owned(), content="[4, 1]")
  assert_eq(layout.patterns[0], (36L, 292L))
  assert_eq(layout.patterns[1], (548L, 804L))
  assert_eq(schedule.steps()[0].pattern, schedule.steps()[1].pattern)
  assert_true(
    prediction_time_bytes(schedule.steps()[0].values[:]) !=
    prediction_time_bytes(schedule.steps()[1].values[:]),
  )
  assert_eq(
    prediction_time_bytes([0.0F, 0.5F, 1.0F]),
    b"\x00\x00\x00\x00\x00\x00\x00\x3f\x00\x00\x80\x3f",
  )
  for budget in [-1L, 0L, 1059L] {
    try prediction_metadata_layout(schedule, budget) catch {
      @device.DeviceError::InvalidArgument(AllocationCreate) => ()
      _ => fail("wrong metadata budget error")
    } noraise {
      _ => fail("metadata budget ignored")
    }
  }
}
