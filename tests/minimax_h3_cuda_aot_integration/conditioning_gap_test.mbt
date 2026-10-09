///|
fn conditioning_plan(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
) -> @joint_diffusion_plan.JointDiffusionPlan {
  let metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant~,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(String::make(64, '6')),
  )
  try! @minimax_h3.build(metadata)
}

///|
fn conditioning_text() -> @joint_diffusion_io_plan.ValidatedTextConditioning {
  try! @joint_diffusion_io_plan.ValidatedTextConditioning::new(
    content_digest=try! @spec.ContentDigest::from_sha256(String::make(64, '7')),
    token_count=137,
  )
}

///|
fn conditioning_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      OrderedKeyframeConditioning(_, _)
      | OrderedReferenceConditioning(_, _, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("ordered conditioning requirement missing")
}

///|
fn conditioning_limits(
  rows? : Int64 = 100000L,
  elements? : Int64 = 10000000L,
) -> @joint_diffusion_conditioning_reference.JointConditioningReferenceLimits {
  try! @joint_diffusion_conditioning_reference.JointConditioningReferenceLimits::new(
    maximum_conditioning_rows=rows,
    maximum_conditioning_elements=elements,
  )
}

///|
fn fl_conditioning_request(
  keyframe_count? : Int = 2,
) -> @joint_diffusion_io_plan.JointDiffusionIoPlan {
  let plan = conditioning_plan(Fl2va)
  let keyframes : Array[@joint_diffusion_io_plan.JointKeyframeConditioning] = [
    @joint_diffusion_io_plan.JointKeyframeConditioning::new(
      position=FirstFrame,
      image=try! @joint_diffusion_io_plan.ValidatedImageInput::new(
        content_digest=try! @spec.ContentDigest::from_sha256(
          String::make(64, '8'),
        ),
        height=768,
        width=1344,
      ),
    ),
  ]
  if keyframe_count == 2 {
    keyframes.push(
      @joint_diffusion_io_plan.JointKeyframeConditioning::new(
        position=LastFrame,
        image=try! @joint_diffusion_io_plan.ValidatedImageInput::new(
          content_digest=try! @spec.ContentDigest::from_sha256(
            String::make(64, '9'),
          ),
          height=768,
          width=1344,
        ),
      ),
    )
  }
  try! @joint_diffusion_io_plan.normalize(
    plan,
    workflow=FirstLastFrameToAudioVideo,
    text=conditioning_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed=1L,
    guidance_scale=1.0,
    keyframes~,
  )
}

///|
fn ref_conditioning_request() -> @joint_diffusion_io_plan.JointDiffusionIoPlan {
  let plan = conditioning_plan(Ref2va)
  try! @joint_diffusion_io_plan.normalize(
    plan,
    workflow=ReferenceToAudioVideo,
    text=conditioning_text(),
    height=768,
    width=1344,
    requested_frames=120,
    inference_steps=50,
    seed=1L,
    guidance_scale=1.0,
    references=[
      ImageReference(
        try! @joint_diffusion_io_plan.ValidatedImageInput::new(
          content_digest=try! @spec.ContentDigest::from_sha256(
            String::make(64, 'a'),
          ),
          height=768,
          width=1344,
        ),
      ),
      VideoReference(
        try! @joint_diffusion_io_plan.ValidatedVideoInput::new(
          content_digest=try! @spec.ContentDigest::from_sha256(
            String::make(64, 'b'),
          ),
          height=768,
          width=1344,
          frame_count=124,
          frames_per_second=24,
        ),
      ),
      AudioReference(
        try! @joint_diffusion_io_plan.ValidatedAudioInput::new(
          content_digest=try! @spec.ContentDigest::from_sha256(
            String::make(64, 'c'),
          ),
          sample_count=160000L,
          sample_rate=32000,
          channel_count=2,
        ),
      ),
    ],
  )
}

///|
test "FL2VA exposes exact keyframe geometry but no AOT authority" {
  let requirements = official_requirements(Fl2va)
  let assessment = @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
    requirements,
    fl_conditioning_request(),
    requirement_ordinal=conditioning_ordinal(requirements),
    limits=conditioning_limits(),
  )
  let geometry = assessment.geometry()
  assert_eq(geometry.operation(), OrderedKeyframeConditioning(2, 24))
  assert_eq(geometry.workflow(), FirstLastFrameToAudioVideo)
  assert_eq(geometry.keyframe_count(), 2)
  assert_eq(geometry.visual_latent_channels(), 24)
  assert_eq(geometry.visual_row_width(), Some(96))
  assert_eq(geometry.visual_rows_per_condition(), Some(1008L))
  assert_eq(geometry.total_visual_conditioning_rows(), Some(2016L))
  assert_eq(geometry.total_visual_conditioning_elements(), Some(193536L))
  assert_eq(assessment.gap_count(), 7)
  assert_eq(assessment.gap(0), Some(DecodedMediaPayloadBuffers))
  assert_eq(assessment.gap(1), Some(VideoVaeEncodeExecution))
  assert_eq(assessment.gap(6), Some(PackedConditioningRowSpans))
  assert_false(assessment.aot_candidate_available())
  inspect(
    assessment.gap_digest().as_hex(),
    content=(
      #|6b5a46a4fc9f5d36234c6302ccaea4a6ad517172b760ba8b6989f38251a02ec0
    ),
  )
  try
    @joint_diffusion_conditioning_reference.require_ordered_conditioning_aot_candidate(
      assessment,
    )
  catch {
    ConditioningSemanticUnavailable(
      OrderedKeyframeConditioning(2, 24),
      DecodedMediaPayloadBuffers
    ) => ()
    _ => fail("unexpected keyframe conditioning gap")
  } noraise {
    _ => fail("keyframe conditioning fabricated AOT authority")
  }
}

///|
test "Ref2VA retains per-reference and soundtrack blockers" {
  let requirements = official_requirements(Ref2va)
  let assessment = @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
    requirements,
    ref_conditioning_request(),
    requirement_ordinal=conditioning_ordinal(requirements),
    limits=conditioning_limits(),
  )
  let geometry = assessment.geometry()
  assert_eq(
    geometry.operation(),
    OrderedReferenceConditioning(9, 3, 3, 12, false),
  )
  assert_eq(geometry.image_reference_count(), 1)
  assert_eq(geometry.video_reference_count(), 1)
  assert_eq(geometry.audio_reference_count(), 1)
  assert_eq(geometry.visual_row_width(), None)
  assert_eq(assessment.gap_count(), 11)
  assert_eq(assessment.gap(4), Some(AudioVaeEncodeExecution))
  assert_eq(assessment.gap(9), Some(PerReferenceEncodedLatentGeometry))
  assert_eq(assessment.gap(10), Some(ReferenceVideoSoundtrackAssociation))
  assert_false(assessment.aot_candidate_available())
  inspect(
    assessment.gap_digest().as_hex(),
    content=(
      #|eca64f142c5cd840294a5b153b8c97558d8e61c585a86d4c49b36e3e7ae760ae
    ),
  )
}

///|
test "conditioning assessment rejects identity ordinal and budget substitutions" {
  let fl = official_requirements(Fl2va)
  try
    @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
      fl,
      ref_conditioning_request(),
      requirement_ordinal=conditioning_ordinal(fl),
      limits=conditioning_limits(),
    )
  catch {
    InvalidRequest => ()
    _ => fail("unexpected foreign conditioning rejection")
  } noraise {
    _ => fail("foreign conditioning identity admitted")
  }
  try
    @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
      fl,
      fl_conditioning_request(),
      requirement_ordinal=0,
      limits=conditioning_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected conditioning ordinal rejection")
  } noraise {
    _ => fail("text requirement was admitted as ordered conditioning")
  }
  try
    @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
      fl,
      fl_conditioning_request(),
      requirement_ordinal=conditioning_ordinal(fl),
      limits=conditioning_limits(rows=2015L),
    )
  catch {
    GeometryBudgetExceeded => ()
    _ => fail("unexpected conditioning budget rejection")
  } noraise {
    _ => fail("oversized conditioning geometry admitted")
  }
  let one = @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
    fl,
    fl_conditioning_request(keyframe_count=1),
    requirement_ordinal=conditioning_ordinal(fl),
    limits=conditioning_limits(),
  )
  let two = @joint_diffusion_conditioning_reference.assess_ordered_conditioning_aot(
    fl,
    fl_conditioning_request(keyframe_count=2),
    requirement_ordinal=conditioning_ordinal(fl),
    limits=conditioning_limits(),
  )
  assert_true(one.gap_digest() != two.gap_digest())
}
