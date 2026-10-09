///|
fn io_content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn io_minimax_plan(
  variant? : @minimax_h3_spec.MiniMaxH3Variant = Fl2va,
  content_seed? : Char = '9',
) -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant~,
        checkpoint_format=DiffusersV1,
      ),
      io_content(content_seed),
    ),
  )
}

///|
fn io_text() -> ValidatedTextConditioning {
  try! ValidatedTextConditioning::new(
    content_digest=io_content('a'),
    token_count=137,
  )
}

///|
fn io_image(seed : Char) -> ValidatedImageInput {
  try! ValidatedImageInput::new(
    content_digest=io_content(seed),
    height=768,
    width=1344,
  )
}

///|
fn io_video(seed : Char) -> ValidatedVideoInput {
  try! ValidatedVideoInput::new(
    content_digest=io_content(seed),
    height=768,
    width=1344,
    frame_count=72,
    frames_per_second=24,
  )
}

///|
fn io_audio(seed : Char) -> ValidatedAudioInput {
  try! ValidatedAudioInput::new(
    content_digest=io_content(seed),
    sample_count=96000L,
    sample_rate=32000,
    channel_count=2,
  )
}

///|
fn io_text_request(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  seed? : Int64 = 42L,
) -> JointDiffusionIoPlan {
  try! normalize(
    plan,
    workflow=TextToAudioVideo,
    text=io_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed~,
    guidance_scale=1.0,
  )
}

///|
test "MiniMax FL2VA projection binds exact resolved media and latent shape" {
  let plan = io_minimax_plan()
  let request = io_text_request(plan)
  assert_eq(request.schema_version(), 1)
  assert_eq(request.identity().content(), plan.content_digest())
  assert_eq(request.identity().plan(), plan.plan_digest())
  assert_eq(request.workflow(), TextToAudioVideo)
  assert_eq(request.text().token_count(), 137)
  assert_eq(request.shape().height(), 768)
  assert_eq(request.shape().width(), 1344)
  assert_eq(request.shape().frames(), 124)
  assert_eq(request.shape().inference_steps(), 50)
  assert_eq(request.shape().video_latent_frames(), 37)
  assert_eq(request.shape().video_latent_height(), 48)
  assert_eq(request.shape().video_latent_width(), 84)
  assert_eq(request.video_latent_channels(), 24)
  assert_eq(request.shape().audio_latent_frames(), 207)
  assert_eq(request.audio_latent_channels(), 32)
  assert_eq(request.shape().audio_samples(), 165333L)
  assert_eq(request.audio_sample_rate(), 32000)
  assert_eq(request.audio_output_channels(), 2)
  assert_eq(request.guidance(), DistilledUnitGuidance)
  assert_eq(request.cancellation(), StageBoundaryCancellation)
  assert_eq(request.stage_count(), plan.stages().length())
}

///|
test "FL2VA keyframes canonicalize by semantic position and bind content" {
  let plan = io_minimax_plan()
  let first = JointKeyframeConditioning::new(
    position=FirstFrame,
    image=io_image('b'),
  )
  let last = JointKeyframeConditioning::new(
    position=LastFrame,
    image=io_image('c'),
  )
  let reversed = try! normalize(
    plan,
    workflow=FirstLastFrameToAudioVideo,
    text=io_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed=7L,
    guidance_scale=1.0,
    keyframes=[last, first],
  )
  let canonical = try! normalize(
    plan,
    workflow=FirstLastFrameToAudioVideo,
    text=io_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed=7L,
    guidance_scale=1.0,
    keyframes=[first, last],
  )
  assert_eq(reversed.keyframes().map(value => value.position()), [
    FirstFrame,
    LastFrame,
  ])
  assert_eq(reversed.request_digest(), canonical.request_digest())
}

///|
test "MiniMax Ref2VA preserves reference order in request identity" {
  let plan = io_minimax_plan(variant=Ref2va)
  let image = JointReferenceConditioning::ImageReference(io_image('b'))
  let video = JointReferenceConditioning::VideoReference(io_video('c'))
  let audio = JointReferenceConditioning::AudioReference(io_audio('d'))
  let first = try! normalize(
    plan,
    workflow=ReferenceToAudioVideo,
    text=io_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed=99L,
    guidance_scale=1.0,
    references=[image, video, audio],
  )
  let reordered = try! normalize(
    plan,
    workflow=ReferenceToAudioVideo,
    text=io_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed=99L,
    guidance_scale=1.0,
    references=[video, image, audio],
  )
  assert_eq(first.references(), [image, video, audio])
  assert_true(first.request_digest() != reordered.request_digest())
}

///|
test "request identity is deterministic and seed and model sensitive" {
  let plan = io_minimax_plan()
  let first = io_text_request(plan)
  assert_eq(first.request_digest(), io_text_request(plan).request_digest())
  assert_true(
    first.request_digest() != io_text_request(plan, seed=43L).request_digest(),
  )
  assert_true(
    first.request_digest() !=
    io_text_request(io_minimax_plan(content_seed='8')).request_digest(),
  )
}

///|
test "workflow and guidance mismatches fail closed" {
  let plan = io_minimax_plan()
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
      references=[ImageReference(io_image('b'))],
    )
  catch {
    WorkflowNotAdmitted => ()
    _ => fail("unexpected workflow rejection")
  } noraise {
    _ => fail("foreign workflow was admitted")
  }
  for hostile in [0.0, 7.5, @double.not_a_number, @double.infinity] {
    try
      normalize(
        plan,
        workflow=TextToAudioVideo,
        text=io_text(),
        height=768,
        width=1344,
        requested_frames=120,
        inference_steps=50,
        seed=0L,
        guidance_scale=hostile,
      )
    catch {
      InvalidGuidancePolicy => ()
      _ => fail("unexpected guidance rejection")
    } noraise {
      _ => fail("hostile guidance was admitted")
    }
  }
}

///|
test "duplicate keyframes and non-normalized media fail closed" {
  let plan = io_minimax_plan()
  let first = JointKeyframeConditioning::new(
    position=FirstFrame,
    image=io_image('b'),
  )
  try
    normalize(
      plan,
      workflow=FirstLastFrameToAudioVideo,
      text=io_text(),
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
      seed=0L,
      guidance_scale=1.0,
      keyframes=[first, first],
    )
  catch {
    InvalidKeyframeConditioning => ()
    _ => fail("unexpected duplicate keyframe rejection")
  } noraise {
    _ => fail("duplicate keyframe position was admitted")
  }
  let wrong_canvas = try! ValidatedImageInput::new(
    content_digest=io_content('c'),
    height=736,
    width=1344,
  )
  try
    normalize(
      plan,
      workflow=FirstLastFrameToAudioVideo,
      text=io_text(),
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
      seed=0L,
      guidance_scale=1.0,
      keyframes=[
        JointKeyframeConditioning::new(position=FirstFrame, image=wrong_canvas),
      ],
    )
  catch {
    InvalidMediaDescriptor => ()
    _ => fail("unexpected media rejection")
  } noraise {
    _ => fail("non-normalized keyframe was admitted")
  }
}

///|
test "Ref2VA count ceilings and audio-only conditioning fail closed" {
  let plan = io_minimax_plan(variant=Ref2va)
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
      references=[AudioReference(io_audio('b'))],
    )
  catch {
    InvalidReferenceConditioning => ()
    _ => fail("unexpected audio-only rejection")
  } noraise {
    _ => fail("audio-only references were admitted")
  }
  let image = JointReferenceConditioning::ImageReference(io_image('c'))
  let references = Array::make(13, image)
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
      references~,
    )
  catch {
    InvalidReferenceConditioning => ()
    _ => fail("unexpected count rejection")
  } noraise {
    _ => fail("reference ceiling was bypassed")
  }
}

///|
test "terminal metadata remains request-bound and exact" {
  let request = io_text_request(io_minimax_plan())
  let success = try! JointDiffusionResultMetadata::succeeded(
    request,
    video_frame_count=124,
    audio_sample_count=165333L,
  )
  assert_eq(success.identity(), request.identity())
  assert_eq(success.request_digest(), request.request_digest())
  assert_eq(success.status(), Succeeded)
  assert_eq(success.completed_stage_count(), request.stage_count())
  assert_eq(success.video_frame_count(), Some(124))
  assert_eq(success.audio_sample_count(), Some(165333L))
  let cancelled = try! JointDiffusionResultMetadata::cancelled(
    request,
    reason=CallerRequested,
    completed_stage_count=3,
  )
  assert_eq(cancelled.status(), Cancelled(CallerRequested))
  assert_eq(cancelled.video_frame_count(), None)
  assert_eq(cancelled.audio_sample_count(), None)
}

///|
test "forged output counts and completed cancellation fail closed" {
  let request = io_text_request(io_minimax_plan())
  try
    JointDiffusionResultMetadata::succeeded(
      request,
      video_frame_count=120,
      audio_sample_count=165333L,
    )
  catch {
    InvalidResultMetadata => ()
    _ => fail("unexpected output rejection")
  } noraise {
    _ => fail("wrong output shape was admitted")
  }
  try
    JointDiffusionResultMetadata::cancelled(
      request,
      reason=RuntimeShutdown,
      completed_stage_count=request.stage_count(),
    )
  catch {
    InvalidCancellationProgress => ()
    _ => fail("unexpected cancellation rejection")
  } noraise {
    _ => fail("completed request was marked cancelled")
  }
}
