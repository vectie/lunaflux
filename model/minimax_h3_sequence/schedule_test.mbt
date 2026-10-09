///|
test "schedule shares order patterns and combines AdaLN modality overrides" {
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
  assert_eq(schedule.steps().length(), 3)
  assert_eq(schedule.patterns().length(), 2)
  assert_eq(schedule.steps()[0].pattern, schedule.steps()[1].pattern)
  assert_true(schedule.steps()[0].values[0] != schedule.steps()[1].values[0])
  assert_true(schedule.steps()[0].pattern != schedule.steps()[2].pattern)
  let first = schedule.patterns()[schedule.steps()[0].pattern]
  assert_eq(first.combined_adaln[0], 3)
  assert_eq(first.combined_adaln[1], 8)
  assert_eq(first.combined_adaln[2], 9)
  assert_eq(first.combined_adaln[3], 2)
  assert_eq(first.combined_adaln[5], 3)
  assert_eq(first.combined_adaln[63], 3)
  let last = schedule.patterns()[schedule.steps()[2].pattern]
  for index in last.indices {
    assert_eq(index, 0)
  }
  assert_eq(last.combined_adaln[0], 0)
  assert_eq(last.combined_adaln[1], 2)
  assert_eq(last.combined_adaln[3], 2)
  assert_eq(last.combined_adaln[63], 0)
  for (step, video, audio) in [(0, 0.2, 0.1), (1, 0.25, 0.15), (2, 0.75, 0.75)] {
    let single = sequence.timesteps(
      video~,
      audio~,
      visual_noise=0.5,
      audio_noise=0.3,
    )
    let pattern = schedule.patterns()[schedule.steps()[step].pattern]
    for row in 0..<64 {
      assert_eq(single.indices()[row], pattern.indices[row])
    }
  }
}

///|
test "schedule index storage budget applies to new patterns only" {
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=4,
    text_rows=1,
    video_rows=2,
    audio_rows=2,
    conditioning=[],
    max_rows=64,
  )
  let shared = sequence.schedule(
    video=[0.2, 0.3, 0.4],
    audio=[0.1, 0.2, 0.3],
    visual_noise=0.0,
    audio_noise=0.0,
    vision_text_rows=[],
    max_steps=3,
    max_index_bytes=512L,
  )
  assert_eq(shared.patterns().length(), 1)
  assert_eq(shared.patterns()[0].combined_adaln[0], 4)
  try
    sequence.schedule(
      video=[0.0],
      audio=[0.0],
      visual_noise=0.0,
      audio_noise=0.0,
      vision_text_rows=[],
      max_steps=1,
      max_index_bytes=511L,
    )
  catch {
    @minimax_h3_sequence.ScheduleError::IndexBudgetExceeded => ()
    _ => fail("wrong budget error")
  } noraise {
    _ => fail("index budget exceeded")
  }
  try
    sequence.schedule(
      video=[0.0],
      audio=[0.0],
      visual_noise=0.0,
      audio_noise=0.0,
      vision_text_rows=[1],
      max_steps=1,
      max_index_bytes=512L,
    )
  catch {
    @minimax_h3_sequence.ScheduleError::InvalidVisionOverride => ()
    _ => fail("wrong override error")
  } noraise {
    _ => fail("non-text override admitted")
  }
}
