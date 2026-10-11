///|
fn publication_variant(
  model : @joint_diffusion_plan.JointDiffusionPlan,
  audio_length : @joint_diffusion_plan.JointAudioOutputLength,
) -> @joint_diffusion_plan.JointDiffusionPlan raise {
  let shapes = model.shapes()
  let publication = @joint_diffusion_plan.JointDiffusionShapeContract::new(
    frames_per_second=shapes.frames_per_second(),
    min_duration_seconds=shapes.min_duration_seconds(),
    max_duration_seconds=shapes.max_duration_seconds(),
    frames_per_chunk=shapes.frames_per_chunk(),
    latents_per_chunk=shapes.latents_per_chunk(),
    canvas_multiple=shapes.canvas_multiple(),
    max_canvas_pixels=shapes.max_canvas_pixels(),
    min_aspect_numerator=shapes.min_aspect_numerator(),
    min_aspect_denominator=shapes.min_aspect_denominator(),
    max_aspect_numerator=shapes.max_aspect_numerator(),
    max_aspect_denominator=shapes.max_aspect_denominator(),
    video_latent_channels=shapes.video_latent_channels(),
    video_spatial_compression=shapes.video_spatial_compression(),
    audio_latent_channels=shapes.audio_latent_channels(),
    audio_latent_streams=shapes.audio_latent_streams(),
    audio_latents_per_second=shapes.audio_latents_per_second(),
    audio_sample_rate=shapes.audio_sample_rate(),
    audio_output_channels=shapes.audio_output_channels(),
    audio_output_length=audio_length,
    audio_vae_latent_dimension=shapes.audio_vae_latent_dimension(),
    audio_vae_decoder_dimension=shapes.audio_vae_decoder_dimension(),
    min_inference_steps=shapes.min_inference_steps(),
    max_inference_steps=shapes.max_inference_steps(),
    sigma_grid=shapes.sigma_grid(),
    video_patch_temporal=shapes.video_patch_temporal(),
    video_patch_height=shapes.video_patch_height(),
    video_patch_width=shapes.video_patch_width(),
  )
  @joint_diffusion_plan.JointDiffusionPlan::new(
    schema_version=model.schema_version(),
    content_digest=model.content_digest(),
    workflows=model.workflows()[:],
    stages=model.stages()[:],
    video_scheduler=model.video_scheduler(),
    audio_scheduler=model.audio_scheduler(),
    shapes=publication,
    conditioning=model.conditioning(),
    denoiser=model.denoiser(),
    projection=model.projection(),
  )
}

///|
test "audio publication rounding cannot reroll initial video or audio noise" {
  for variant in [@minimax_h3_spec.MiniMaxH3Variant::Fl2va, Ref2va] {
    let original = @minimax_h3.build(
      @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
        @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
          variant~,
          checkpoint_format=DiffusersV1,
        ),
        @spec.ContentDigest::from_sha256(String::make(64, '7')),
      ),
    )
    let nominal = publication_variant(original, FrameDuration)
    assert_true(original.plan_digest() != nominal.plan_digest())
    let plans = [original, nominal].map(model => {
      let request = model
        .shapes()
        .resolve_request(
          height=32,
          width=32,
          requested_frames=120,
          inference_steps=5,
        )
      let prepared = @joint_diffusion_loop.PreparedPlan::new(
        model,
        request,
        max_latent_elements=1048576L,
      )
      @joint_diffusion_loop.LatentStoragePlan::new(prepared, max_bytes=1048576L)
    })
    let latents : Array[@joint_diffusion_plan.JointDiffusionLatentKind] = [
      VideoLatent,
      AudioLatent,
    ]
    for seed in [0L, 17L, -1L] {
      let first = plans[0].noise(seed~, max_chunk_elements=1048576)
      let second = plans[1].noise(seed~, max_chunk_elements=1048576)
      for latent in latents {
        let count = first.element_count(latent).to_int()
        assert_eq(second.element_count(latent), count.to_int64())
        assert_eq(
          first.chunk(latent, start=0L, count~),
          second.chunk(latent, start=0L, count~),
        )
        let other = plans[1].noise(seed=seed ^ 1L, max_chunk_elements=1048576)
        assert_true(
          first.chunk(latent, start=0L, count=13) !=
          other.chunk(latent, start=0L, count=13),
        )
      }
    }
  }
}
