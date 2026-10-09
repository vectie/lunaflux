///|
fn content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn minimax_plan(
  variant? : @minimax_h3_spec.MiniMaxH3Variant = Fl2va,
) -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant~,
        checkpoint_format=DiffusersV1,
      ),
      content('9'),
    ),
  )
}

///|
fn tiny_plan() -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=1,
    content_digest=content('8'),
    workflows=[TextToAudioVideo],
    stages=[
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=ResolveRequest,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=EncodeTextConditioning,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=InitializeJointLatents,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=JointAudioVideoDenoise,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeVideo,
        presence=Required,
      ),
      @joint_diffusion_plan.JointDiffusionStage::new(
        kind=DecodeAudio,
        presence=Required,
      ),
    ],
    video_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=VideoLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=2,
      shift_denominator=1,
    ),
    audio_scheduler=@joint_diffusion_plan.JointDiffusionScheduler::new(
      latent=AudioLatent,
      kind=RectifiedFlowEuler,
      shift_numerator=1,
      shift_denominator=1,
    ),
    shapes=@joint_diffusion_plan.JointDiffusionShapeContract::new(
      frames_per_second=4,
      min_duration_seconds=1,
      max_duration_seconds=2,
      frames_per_chunk=3,
      latents_per_chunk=1,
      canvas_multiple=4,
      max_canvas_pixels=16L,
      min_aspect_numerator=1,
      min_aspect_denominator=1,
      max_aspect_numerator=1,
      max_aspect_denominator=1,
      video_latent_channels=2,
      video_spatial_compression=2,
      audio_latent_channels=2,
      audio_latents_per_second=2,
      audio_sample_rate=8,
      audio_output_channels=1,
      audio_vae_latent_dimension=4,
      audio_vae_decoder_dimension=2,
      min_inference_steps=2,
      max_inference_steps=8,
      video_patch_temporal=1,
      video_patch_height=2,
      video_patch_width=2,
    ),
    conditioning=@joint_diffusion_plan.JointDiffusionConditioningLimits::new(
      max_keyframes=0,
      max_image_references=0,
      max_video_references=0,
      max_audio_references=0,
      max_total_references=0,
      allow_audio_only_references=false,
    ),
    projection=@joint_diffusion_plan.JointDiffusionProjectionContract::mixed_precision_float32_heads(),
    denoiser=@joint_diffusion_plan.JointDiffusionDenoiserShape::new(
      hidden_size=4,
      layer_count=1,
      refiner_layer_count=0,
      attention_head_count=1,
      attention_head_dimension=4,
      feed_forward_dimension=8,
      text_conditioning_dimension=4,
      timestep_frequency_dimension=2,
    ),
  )
}

///|
fn tiny_request(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
  steps? : Int = 5,
) -> @joint_diffusion_plan.JointDiffusionRequestShape {
  try! plan
  .shapes()
  .resolve_request(height=4, width=4, requested_frames=4, inference_steps=steps)
}

///|
fn close(actual : Double, expected : Double) -> Bool {
  (actual - expected).abs() <= 0.0000000000001
}

///|
test "independent shifted Rectified Flow fixtures match both MiniMax modalities" {
  let plan = minimax_plan()
  let request = plan
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
  let video = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
    plan,
    request,
    VideoLatent,
  )
  let audio = @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
    plan,
    request,
    AudioLatent,
  )
  let video_fixture = [
    1.0, 0.973008024641323, 0.923218814848974, 0.8006385950908003, 0.011869436201780425,
    0.0,
  ]
  let audio_fixture = [
    1.0, 0.9001199760047992, 0.7503748125937031, 0.5009990009990011, 0.002994011976047907,
    0.0,
  ]
  for index in 0..<6 {
    assert_true(close(video.sigma(index).unwrap(), video_fixture[index]))
    assert_true(close(audio.sigma(index).unwrap(), audio_fixture[index]))
    if index < 5 {
      assert_true(
        close(video.timestep(index).unwrap(), video_fixture[index] * 1000.0),
      )
    } else {
      assert_eq(video.timestep(index), None)
    }
  }
  assert_eq(video.evaluation_count(), 5)
  assert_eq(audio.point_count(), 6)
}

///|
test "packed rows use patch geometry and split recombines exactly" {
  let plan = tiny_plan()
  let layout = @joint_diffusion_reference.JointPackedRowLayout::new(
    plan,
    tiny_request(plan),
    conditioning_rows=2,
  )
  assert_eq(layout.hidden_width(), 4)
  assert_eq(layout.conditioning_rows(), 2)
  assert_eq(layout.video_rows(), 3)
  assert_eq(layout.audio_rows(), 2)
  assert_eq(layout.total_rows(), 7)
  assert_eq(layout.row_start(ConditioningRows), 0)
  assert_eq(layout.row_start(VideoRows), 2)
  assert_eq(layout.row_start(AudioRows), 5)
  assert_eq(layout.row_count(VideoRows), 3)
  assert_eq(layout.video_input_width(), 8)
  assert_eq(layout.audio_input_width(), 2)
  let conditioning = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
  let video = [
    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
  ]
  let audio = [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0]
  let packed = @joint_diffusion_reference.recombine_projected_rows(
    layout,
    conditioning,
    video,
    audio,
    max_elements=28,
  )
  let split = @joint_diffusion_reference.split_projected_rows(packed)
  assert_eq(split.conditioning(), ReadOnlyArray::from_array(conditioning))
  assert_eq(split.video(), ReadOnlyArray::from_array(video))
  assert_eq(split.audio(), ReadOnlyArray::from_array(audio))
  assert_eq(packed.values()[8], 10.0)
  assert_eq(packed.values()[20], 30.0)
}

///|
test "counter-based streams are repeatable disjoint and bounded" {
  let plan = tiny_plan()
  let rng = @joint_diffusion_reference.CounterBasedJointLatentRng::new(
    plan,
    tiny_request(plan),
    seed=42L,
    conditioning_draw_limit=3L,
  )
  let conditioning = rng.normal(ConditioningStream, 0L)
  let video = rng.normal(VideoLatentStream, 0L)
  let audio = rng.normal(AudioLatentStream, 0L)
  assert_eq(conditioning, rng.normal(ConditioningStream, 0L))
  assert_true(conditioning != video)
  assert_true(video != audio)
  assert_eq(rng.draw_limit(ConditioningStream), 3L)
  assert_eq(rng.draw_limit(VideoLatentStream), 24L)
  assert_eq(rng.draw_limit(AudioLatentStream), 4L)
  assert_true(close(conditioning, 0.626064151676309))
  assert_true(close(video, -1.2409333736868997))
  assert_true(close(audio, -0.5518632908582728))
  try rng.normal(AudioLatentStream, 4L) catch {
    InvalidRandomCounter(AudioLatentStream) => ()
    _ => fail("unexpected audio counter error")
  } noraise {
    _ => fail("out-of-range audio counter was admitted")
  }
}

///|
test "stage machine enforces order optional skip and cancellation" {
  let plan = minimax_plan()
  let initial = @joint_diffusion_reference.JointDiffusionStageMachine::start(
    plan,
  )
  try initial.advance(EncodeTextConditioning) catch {
    StageMismatch(expected=ResolveRequest, actual=EncodeTextConditioning) => ()
    _ => fail("unexpected stage mismatch")
  } noraise {
    _ => fail("out-of-order stage was admitted")
  }
  let after_text = initial
    .advance(ResolveRequest)
    .advance(EncodeTextConditioning)
  assert_eq(after_text.next_stage().unwrap().kind(), EncodeKeyframeConditioning)
  let after_optional = after_text.skip_optional()
  try after_optional.skip_optional() catch {
    RequiredStageCannotBeSkipped(InitializeJointLatents) => ()
    _ => fail("unexpected required-stage skip error")
  } noraise {
    _ => fail("required stage was skipped")
  }
  let cancelled = after_optional.cancel()
  assert_true(cancelled.is_cancelled())
  assert_eq(cancelled.next_stage(), None)
  try cancelled.advance(InitializeJointLatents) catch {
    StageMachineCancelled => ()
    _ => fail("unexpected cancellation result")
  } noraise {
    _ => fail("cancelled pipeline advanced")
  }
}

///|
test "stage machine reaches one terminal completion and cannot restart" {
  let complete = @joint_diffusion_reference.JointDiffusionStageMachine::start(
      tiny_plan(),
    )
    .advance(ResolveRequest)
    .advance(EncodeTextConditioning)
    .advance(InitializeJointLatents)
    .advance(JointAudioVideoDenoise)
    .advance(DecodeVideo)
    .advance(DecodeAudio)
  assert_true(complete.is_complete())
  assert_eq(complete.completed_stage_count(), 6)
  assert_eq(complete.next_stage(), None)
  try complete.advance(DecodeAudio) catch {
    StageMachineComplete => ()
    _ => fail("unexpected completed-machine advance result")
  } noraise {
    _ => fail("completed machine restarted")
  }
  try complete.cancel() catch {
    CompletedMachineCannotBeCancelled => ()
    _ => fail("unexpected completed-machine cancellation result")
  } noraise {
    _ => fail("completed machine was converted into cancellation")
  }
}

///|
test "hostile layout and schedule inputs fail before allocation" {
  let first = tiny_plan()
  let second = minimax_plan()
  let foreign = second
    .shapes()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
  try
    @joint_diffusion_reference.build_shifted_rectified_flow_schedule(
      first,
      foreign,
      VideoLatent,
    )
  catch {
    InvalidRequestShape => ()
    _ => fail("unexpected foreign-shape schedule error")
  } noraise {
    _ => fail("foreign request shape was admitted")
  }
  let layout = @joint_diffusion_reference.JointPackedRowLayout::new(
    first,
    tiny_request(first),
    conditioning_rows=0,
  )
  try
    @joint_diffusion_reference.recombine_projected_rows(
      layout,
      [],
      [1.0],
      [],
      max_elements=100,
    )
  catch {
    InvalidProjectedRowCount(VideoRows) => ()
    _ => fail("unexpected projected-row shape error")
  } noraise {
    _ => fail("short projected rows were admitted")
  }
}
