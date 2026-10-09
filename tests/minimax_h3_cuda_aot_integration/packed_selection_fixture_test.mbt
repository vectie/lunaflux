///|
fn packed_selection_fixture_with_io(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
) -> (
  @joint_diffusion_io_plan.JointDiffusionIoPlan,
  @joint_diffusion_io_plan.JointPackedTimestepSelection,
  @joint_diffusion_io_plan.JointPackedModalitySelection,
  @joint_diffusion_io_plan.JointDistinctTimestepValues,
) {
  let metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant~,
      checkpoint_format=DiffusersV1,
    ),
    try! @spec.ContentDigest::from_sha256(String::make(64, '6')),
  )
  let plan = try! @minimax_h3.build(metadata)
  let text = try! @joint_diffusion_io_plan.ValidatedTextConditioning::new(
    content_digest=try! @spec.ContentDigest::from_sha256(String::make(64, '7')),
    token_count=137,
  )
  let io = match variant {
    Fl2va =>
      try! @joint_diffusion_io_plan.normalize(
        plan,
        workflow=TextToAudioVideo,
        text~,
        height=768,
        width=1344,
        requested_frames=120,
        inference_steps=50,
        seed=1L,
        guidance_scale=1.0,
      )
    Ref2va =>
      try! @joint_diffusion_io_plan.normalize(
        plan,
        workflow=ReferenceToAudioVideo,
        text~,
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
                String::make(64, '8'),
              ),
              height=768,
              width=1344,
            ),
          ),
        ],
      )
  }
  let video_rows = 37296
  let audio_rows = 414
  let packed_rows = video_rows + audio_rows + 137
  let timestep = try! @joint_diffusion_io_plan.JointPackedTimestepSelection::new(
    io,
    evaluation_index=0,
    distinct_timestep_count=3,
    indices=Array::make(packed_rows, 0),
  )
  let modality = try! @joint_diffusion_io_plan.JointPackedModalitySelection::new(
    io,
    timestep,
    video_indices=Array::makei(video_rows, index => index),
    audio_indices=Array::makei(audio_rows, index => video_rows + index),
  )
  let values = try! @joint_diffusion_io_plan.JointDistinctTimestepValues::new(
    io,
    timestep,
    values=[0.0F, 0.5F, 1.0F],
  )
  (io, timestep, modality, values)
}

///|
fn packed_selection_fixture(
  variant : @minimax_h3_spec.MiniMaxH3Variant,
) -> (
  @joint_diffusion_io_plan.JointPackedTimestepSelection,
  @joint_diffusion_io_plan.JointPackedModalitySelection,
  @joint_diffusion_io_plan.JointDistinctTimestepValues,
) {
  let (_, timestep, modality, values) = packed_selection_fixture_with_io(
    variant,
  )
  (timestep, modality, values)
}
