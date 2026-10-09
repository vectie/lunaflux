///|
test "official request binds 37296 video rows and 414 stereo audio rows" {
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant=Fl2va,
        checkpoint_format=DiffusersV1,
      ),
      @spec.ContentDigest::from_sha256(String::make(64, '7')),
    ),
  )
  let request = model
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
  let plan = @minimax_h3_sequence.Sequence::from_request(
    model,
    request,
    text_rows=3,
    conditioning=[Image(1008), Image(1008)],
    max_rows=39744,
  )
  assert_eq(plan.assembly().input_rows(Video), 37296)
  assert_eq(plan.assembly().input_rows(Audio), 414)
  assert_eq(plan.assembly().input_rows(Conditioning), 2019)
  assert_eq(plan.used_rows(), 39729)
  assert_eq(plan.assembly().rows().length(), 39744)
}

///|
test "FL2VA text keyframe audio video order includes final padding" {
  let plan = @minimax_h3_sequence.Sequence::new(
    hidden_width=4,
    text_rows=3,
    video_rows=8,
    audio_rows=4,
    conditioning=[Image(2), Image(2)],
    max_rows=64,
  )
  assert_eq(plan.used_rows(), 19)
  assert_eq(plan.assembly().rows().length(), 64)
  let segments = plan.segments()
  assert_true(segments[0].kind is Text)
  assert_true(segments[1].kind is ConditionVideo)
  assert_true(segments[3].kind is TargetAudio)
  assert_true(segments[4].kind is TargetVideo)
  assert_eq(segments[3].start, 7)
  assert_eq(segments[4].start, 11)
  let rows = plan.assembly().rows()
  assert_true(rows[7].source is Audio)
  assert_eq(rows[7].index, 0)
  assert_true(rows[11].source is Video)
  assert_eq(rows[11].index, 0)
  for row in rows[19:] {
    assert_true(row.source is Padding)
  }
}

///|
test "Ref2VA retains block order and paired reference audio precedes video" {
  let plan = @minimax_h3_sequence.Sequence::new(
    hidden_width=8,
    text_rows=2,
    video_rows=5,
    audio_rows=4,
    conditioning=[Image(3), Video(audio_rows=4, video_rows=6), Audio(2)],
    max_rows=64,
  )
  let segments = plan.segments()
  assert_eq(segments.length(), 7)
  assert_true(segments[2].kind is ConditionAudio)
  assert_true(segments[3].kind is ConditionVideo)
  assert_eq(segments[2].start, 5)
  assert_eq(segments[3].start, 9)
  assert_eq(segments[4].start, 15)
  assert_eq(segments[5].start, 17)
  assert_eq(segments[6].start, 21)
  assert_eq(plan.used_rows(), 26)
  assert_eq(plan.assembly().input_rows(Conditioning), 17)
}

///|
test "row ceiling includes padding and invalid conditioning is rejected" {
  try
    @minimax_h3_sequence.Sequence::new(
      hidden_width=4,
      text_rows=1,
      video_rows=1,
      audio_rows=1,
      conditioning=[],
      max_rows=63,
    )
  catch {
    @minimax_h3_sequence.SequenceError::RowLimitExceeded => ()
    _ => fail("wrong row ceiling error")
  } noraise {
    _ => fail("padding exceeded budget")
  }
  for
    block in [
      @minimax_h3_sequence.ConditionBlock::Image(0),
      Audio(-1),
      Video(audio_rows=0, video_rows=0),
    ] {
    try
      @minimax_h3_sequence.Sequence::new(
        hidden_width=4,
        text_rows=1,
        video_rows=1,
        audio_rows=1,
        conditioning=[block],
        max_rows=64,
      )
    catch {
      @minimax_h3_sequence.SequenceError::InvalidConditioning => ()
      _ => fail("wrong conditioning error")
    } noraise {
      _ => fail("invalid conditioning accepted")
    }
  }
}
