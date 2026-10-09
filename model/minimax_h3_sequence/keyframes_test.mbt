///|
test "FL2VA anchors preserve order without advancing target origin" {
  let grid = @minimax_h3_sequence.VideoGrid::new(
    frames=2,
    height=4,
    width=4,
    max_rows=64,
  )
  let prepared = @minimax_h3_sequence.PositionedSequence::keyframes(
    grid,
    hidden_width=4,
    text_rows=2,
    audio_frames=2,
    anchors=[Last, First],
    max_rows=64,
  )
  let p = prepared.positions()
  assert_true((p[2].time - (2.0 + 20.0 / 3.0)).abs() < 0.000000000001)
  assert_eq(p[6].time, 2.0)
  assert_eq(p[10].time, 2.0)
  assert_eq(p[14].time, 2.0)
  assert_eq(p[3].width, 16.0)
  assert_eq(prepared.sequence().used_rows(), 22)
  try
    @minimax_h3_sequence.PositionedSequence::keyframes(
      grid,
      hidden_width=4,
      text_rows=2,
      audio_frames=2,
      anchors=[First, First],
      max_rows=64,
    )
  catch {
    @minimax_h3_sequence.SequenceError::InvalidConditioning => ()
    _ => fail("unexpected anchor error")
  } noraise {
    _ => fail("duplicate keyframe admitted")
  }
}
