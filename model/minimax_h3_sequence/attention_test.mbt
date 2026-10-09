///|
test "joint visibility isolates padding but not live modalities" {
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=4,
    text_rows=3,
    video_rows=4,
    audio_rows=2,
    conditioning=[Image(2), Video(audio_rows=2, video_rows=3)],
    max_rows=64,
  )
  let documents = sequence.attention_documents()
  assert_eq(documents.length(), 2)
  assert_eq(documents[0].start, 0)
  assert_eq(documents[0].count, 16)
  assert_eq(documents[1].start, 16)
  assert_eq(documents[1].count, 48)
  assert_eq(sequence.attention_boundaries().length(), 3)
  assert_eq(sequence.attention_boundaries()[1], 16)
  assert_eq(sequence.attention_boundaries()[2], 64)
  // Every real segment belongs to the same bidirectional attention document.
  for segment in sequence.segments() {
    assert_true(segment.start >= documents[0].start)
    assert_true(segment.start + segment.count <= documents[0].count)
  }
  assert_eq(sequence.refiner_document().start, 0)
  assert_eq(sequence.refiner_document().count, 3)
  let aligned = @minimax_h3_sequence.Sequence::new(
    hidden_width=4,
    text_rows=3,
    video_rows=59,
    audio_rows=2,
    conditioning=[],
    max_rows=64,
  )
  assert_eq(aligned.attention_documents().length(), 1)
  assert_eq(aligned.attention_boundaries().length(), 2)
  assert_eq(aligned.attention_documents()[0].count, 64)
}
