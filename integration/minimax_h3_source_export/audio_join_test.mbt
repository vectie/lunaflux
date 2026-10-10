///|
test "checkpoint audio joint geometry and bridge budget use actual rounded duration" {
  let metadata = @spec.MiniMaxH3ModelMetadata::from_verified_content(
    @spec.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    @identity.ContentDigest::from_sha256(String::make(64, '7')),
  )
  let model = @h3.build(metadata)
  let audio = audio_config()
  for frames in [120, 240] {
    let shape = model
      .shapes()
      .resolve_request(
        height=32,
        width=32,
        requested_frames=frames,
        inference_steps=5,
      )
    let plan = @loop.PreparedPlan::new(
      model,
      shape,
      max_latent_elements=100000000L,
    )
    let sources = @block.build_audio_decoder(
      @vae.bind_audio_decoder_upsample_stages(metadata, audio, shape)[:],
    )
    let decoder = @audio_checkpoint.CheckpointAudioDecoder::new(
      sources,
      maximum_workspace_bytes=2147483648L,
    )
    // Pure startup sizing must work before allocating or preparing a device.
    assert_true(decoder.matches_plan(plan))
    assert_eq(
      decoder.joint_workspace_bytes(),
      decoder.memory().total_bytes + decoder.memory().input_bytes,
    )
    let different = model
      .shapes()
      .resolve_request(
        height=32,
        width=32,
        requested_frames=if frames == 120 { 240 } else { 120 },
        inference_steps=5,
      )
    assert_false(
      decoder.matches_plan(
        @loop.PreparedPlan::new(
          model,
          different,
          max_latent_elements=100000000L,
        ),
      ),
    )
    decoder.close()
    decoder.close()
  }
}
