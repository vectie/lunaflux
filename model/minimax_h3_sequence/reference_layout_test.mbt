///|
test "reference VAE layouts preserve patch and stereo channel order" {
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant=Fl2va,
        checkpoint_format=DiffusersV1,
      ),
      @spec.ContentDigest::from_sha256(String::make(64, '7')),
    ),
  )
  let video = @minimax_h3_sequence.reference_video_layout(
    model,
    frames=1,
    height=2,
    width=4,
    expected_rows=2,
    max_elements=192L,
  )
  assert_eq(video.element_count(), 192L)
  for i, expected in [0L, 1L, 4L, 5L, 8L] {
    assert_eq(video.source_index(i.to_int64()), expected)
  }
  assert_eq(video.source_index(96L), 2L)
  let inverse = video.inverse()
  for i in 0..<192 {
    assert_eq(
      video.source_index(inverse.source_index(i.to_int64())),
      i.to_int64(),
    )
  }
  let audio = @minimax_h3_sequence.reference_audio_layout(
    model,
    frames=3,
    expected_rows=6,
    max_elements=192L,
  )
  assert_eq(audio.source_index(1L), 3L)
  assert_eq(audio.source_index(32L), 1L)
  assert_eq(audio.source_index(96L), 96L)
  for (height, rows, limit) in [(3, 2, 192L), (2, 3, 192L), (2, 2, 191L)] {
    try
      @minimax_h3_sequence.reference_video_layout(
        model,
        frames=1,
        height~,
        width=4,
        expected_rows=rows,
        max_elements=limit,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid reference layout accepted")
    }
  }
  try
    @minimax_h3_sequence.reference_audio_layout(
      model,
      frames=3,
      expected_rows=3,
      max_elements=192L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("mono row count accepted for stereo")
  }
}
