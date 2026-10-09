///|
test "mixed Ref2VA coordinates retain block origins stereo extremes and padding" {
  let target = @minimax_h3_sequence.VideoGrid::new(
    frames=2,
    height=4,
    width=4,
    max_rows=64,
  )
  let image = @minimax_h3_sequence.VideoGrid::new(
    frames=1,
    height=4,
    width=4,
    max_rows=64,
  )
  let prepared = @minimax_h3_sequence.PositionedSequence::references(
    target,
    hidden_width=4,
    text_rows=2,
    audio_frames=2,
    references=[ImageGrid(image), AudioGrid(2), VideoAudioGrid(target, 1)],
    max_rows=64,
  )
  let p = prepared.positions()
  assert_eq(p.length(), 64)
  assert_eq(prepared.sequence().used_rows(), 32)
  assert_eq(p[1].time, 1.0)
  assert_eq(p[2].time, 2.0)
  assert_eq(p[3].width, 16.0)
  assert_eq(p[4].height, 16.0)
  assert_eq(p[6].time, 3.0)
  assert_eq(p[8].time, 3.0)
  assert_eq(p[6].width, 0.0)
  assert_eq(p[8].width, 16.0)
  assert_eq(p[10].time, 5.0)
  assert_eq(p[11].width, 16.0)
  assert_eq(p[12].time, 5.0)
  assert_true((p[16].time - (5.0 + 5.0 / 3.0)).abs() < 0.000000000001)
  assert_true((p[20].time - (5.0 + 25.0 / 3.0)).abs() < 0.000000000001)
  assert_eq(p[20].time, p[24].time)
  for row in p[32:] {
    assert_eq(row.time, 0.0)
    assert_eq(row.height, 0.0)
    assert_eq(row.width, 0.0)
  }
}

///|
test "rectangular spatial grid excludes endpoint and temporal cadence resets every five" {
  let target = @minimax_h3_sequence.VideoGrid::new(
    frames=6,
    height=2,
    width=8,
    max_rows=64,
  )
  let prepared = @minimax_h3_sequence.PositionedSequence::references(
    target,
    hidden_width=4,
    text_rows=1,
    audio_frames=1,
    references=[],
    max_rows=64,
  )
  let p = prepared.positions()
  assert_eq(p[1].width, -16.0)
  assert_eq(p[2].width, 32.0)
  for
    frame, offset in [
      0.0,
      5.0 / 3.0,
      25.0 / 3.0,
      45.0 / 3.0,
      65.0 / 3.0,
      85.0 / 3.0,
    ] {
    let row = 3 + frame * 4
    assert_true((p[row].time - (1.0 + offset)).abs() < 0.000000000001)
    assert_eq(p[row].height, 8.0)
    assert_eq(p[row].width, -16.0)
    assert_eq(p[row + 3].width, 32.0)
  }
  try
    @minimax_h3_sequence.VideoGrid::new(
      frames=1,
      height=3,
      width=4,
      max_rows=64,
    )
  catch {
    @minimax_h3_sequence.SequenceError::InvalidShape => ()
    _ => fail("wrong grid error")
  } noraise {
    _ => fail("odd patch dimension accepted")
  }
}
