///|
test "rotary coefficients concatenate axes repeat half and preserve padding identity" {
  let grid = @minimax_h3_sequence.VideoGrid::new(
    frames=1,
    height=4,
    width=4,
    max_rows=64,
  )
  let sequence = @minimax_h3_sequence.PositionedSequence::references(
    grid,
    hidden_width=4,
    text_rows=2,
    audio_frames=1,
    references=[],
    max_rows=64,
  )
  let tables = sequence.rotary_tables([1.0F, 0.5F], max_elements=1536L)
  assert_eq(tables.width(), 12)
  assert_eq(tables.cosine().length(), 768)
  for row in 0..<64 {
    for lane in 0..<6 {
      assert_eq(
        tables.cosine()[row * 12 + lane],
        tables.cosine()[row * 12 + lane + 6],
      )
      assert_eq(
        tables.sine()[row * 12 + lane],
        tables.sine()[row * 12 + lane + 6],
      )
    }
  }
  assert_eq(tables.cosine()[12], Float::from_double(@math.cos(1.0)))
  assert_eq(tables.cosine()[13], Float::from_double(@math.cos(0.5)))
  assert_eq(tables.cosine()[14], 1.0F)
  assert_eq(tables.sine()[14], 0.0F)
  for index in (8 * 12)..<768 {
    assert_eq(tables.cosine()[index], 1.0F)
    assert_eq(tables.sine()[index], 0.0F)
  }
  try sequence.rotary_tables([1.0F, 0.5F], max_elements=1535L) catch {
    @minimax_h3_sequence.RotaryError::ElementLimitExceeded => ()
    _ => fail("wrong rotary capacity error")
  } noraise {
    _ => fail("short table capacity accepted")
  }
  try sequence.rotary_tables([0.0F], max_elements=1536L) catch {
    @minimax_h3_sequence.RotaryError::InvalidFrequency => ()
    _ => fail("wrong frequency error")
  } noraise {
    _ => fail("invalid frequency admitted")
  }
}
