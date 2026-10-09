///|
test "output duration and inference-step ceilings fail closed" {
  let plan = io_minimax_plan()
  for hostile in [(361, 50), (120, 1), (120, 1001)] {
    let (frames, steps) = hostile
    try
      normalize(
        plan,
        workflow=TextToAudioVideo,
        text=io_text(),
        height=768,
        width=1344,
        requested_frames=frames,
        inference_steps=steps,
        seed=0L,
        guidance_scale=1.0,
      )
    catch {
      InvalidOutputShape => ()
      _ => fail("unexpected output ceiling rejection")
    } noraise {
      _ => fail("hostile output shape was admitted")
    }
  }
}

///|
test "Ref2VA rejects reference media beyond normalized request bounds" {
  let plan = io_minimax_plan(variant=Ref2va)
  let oversized_video = try! ValidatedVideoInput::new(
    content_digest=io_content('b'),
    height=768,
    width=1344,
    frame_count=125,
    frames_per_second=24,
  )
  try
    normalize(
      plan,
      workflow=ReferenceToAudioVideo,
      text=io_text(),
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
      seed=0L,
      guidance_scale=1.0,
      references=[
        ImageReference(io_image('c')),
        VideoReference(oversized_video),
      ],
    )
  catch {
    InvalidMediaDescriptor => ()
    _ => fail("unexpected oversized-video rejection")
  } noraise {
    _ => fail("oversized reference video was admitted")
  }
  let wrong_rate = try! ValidatedAudioInput::new(
    content_digest=io_content('d'),
    sample_count=96000L,
    sample_rate=48000,
    channel_count=2,
  )
  try
    normalize(
      plan,
      workflow=ReferenceToAudioVideo,
      text=io_text(),
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
      seed=0L,
      guidance_scale=1.0,
      references=[ImageReference(io_image('c')), AudioReference(wrong_rate)],
    )
  catch {
    InvalidMediaDescriptor => ()
    _ => fail("unexpected audio-rate rejection")
  } noraise {
    _ => fail("non-normalized audio reference was admitted")
  }
}

///|
test "validated descriptors remain bounded at construction" {
  try
    ValidatedTextConditioning::new(
      content_digest=io_content('a'),
      token_count=0,
    )
  catch {
    InvalidTextConditioning => ()
    _ => fail("unexpected text descriptor rejection")
  } noraise {
    _ => fail("empty text conditioning was admitted")
  }
  try
    ValidatedAudioInput::new(
      content_digest=io_content('b'),
      sample_count=2147483648L,
      sample_rate=32000,
      channel_count=2,
    )
  catch {
    InvalidMediaDescriptor => ()
    _ => fail("unexpected audio descriptor rejection")
  } noraise {
    _ => fail("unbounded audio descriptor was admitted")
  }
}
