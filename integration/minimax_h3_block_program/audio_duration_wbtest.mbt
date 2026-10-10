///|
test "124-frame request retains rounded decoder duration rather than requested sample count" {
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
      requested_frames=124,
      inference_steps=5,
    )
  assert_eq(request.frames(), 124)
  assert_eq(request.audio_latent_frames(), 207)
  assert_eq(request.audio_samples(), 165333L)
  let actual_samples = request.audio_latent_frames() * 800
  assert_eq(actual_samples, 165600)
  assert_eq(actual_samples.to_int64(), request.audio_output_samples())
  assert_true(actual_samples.to_int64() != request.audio_samples())
  assert_true(
    audio_join_geometry(
      request.audio_latent_frames(),
      2,
      actual_samples,
      actual_samples.to_int64() * 8L,
    ),
  )
  assert_false(
    audio_join_geometry(
      request.audio_latent_frames(),
      2,
      request.audio_samples().to_int(),
      request.audio_samples() * 8L,
    ),
  )
  assert_false(
    audio_join_geometry(
      request.audio_latent_frames(),
      1,
      actual_samples,
      actual_samples.to_int64() * 4L,
    ),
  )
  assert_false(
    audio_join_geometry(
      request.audio_latent_frames(),
      2,
      actual_samples,
      actual_samples.to_int64() * 8L - 4L,
    ),
  )
  let decoder = AudioDecodeRequest::new()
  try decoder.waveform_samples() catch {
    _ => ()
  } noraise {
    _ => fail("unprepared waveform geometry")
  }
}
