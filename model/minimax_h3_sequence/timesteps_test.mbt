///|
test "timestep candidates sort collapse and preserve text padding semantics" {
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=4,
    text_rows=1,
    video_rows=2,
    audio_rows=2,
    conditioning=[Video(audio_rows=1, video_rows=1)],
    max_rows=64,
  )
  let times = sequence.timesteps(
    video=0.2,
    audio=0.1,
    visual_noise=0.5,
    audio_noise=0.3,
  )
  assert_eq(times.values().length(), 4)
  for i, expected in [0.1F, 0.2F, 0.3F, 0.5F] {
    assert_eq(times.values()[i], expected)
  }
  assert_eq(times.indices()[0], 1)
  assert_eq(times.indices()[1], 2)
  assert_eq(times.indices()[2], 3)
  assert_eq(times.indices()[3], 0)
  assert_eq(times.indices()[5], 1)
  assert_eq(times.indices()[63], 1)
  let collapsed = sequence.timesteps(
    video=0.75,
    audio=0.75000000001,
    visual_noise=0.1,
    audio_noise=0.2,
  )
  assert_eq(collapsed.values().length(), 1)
  for index in collapsed.indices() {
    assert_eq(index, 0)
  }
  for invalid in [-0.1, 1.1, 0.0 / 0.0, 1.0 / 0.0] {
    try
      sequence.timesteps(
        video=invalid,
        audio=0.0,
        visual_noise=0.0,
        audio_noise=0.0,
      )
    catch {
      @minimax_h3_sequence.TimestepError::InvalidTime => ()
    } noraise {
      _ => fail("invalid timestep accepted")
    }
  }
}
