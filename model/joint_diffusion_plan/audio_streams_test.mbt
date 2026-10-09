///|
test "audio latent stream count is independent of output channels and bounded" {
  for streams in [1, 2, 3] {
    let shapes = test_shapes(streams~)
    let request = shapes.resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
    assert_eq(shapes.audio_output_channels(), 2)
    assert_eq(shapes.audio_latent_streams(), streams)
    assert_eq(request.audio_latent_frames(), 207)
    assert_eq(request.audio_token_rows(), 207 * streams)
  }
  let huge = test_shapes(streams=2147483647)
  try
    huge.resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=5,
    )
  catch {
    @joint_diffusion_plan.JointDiffusionPlanError::InvalidFrameCount => ()
    _ => fail("wrong audio row overflow error")
  } noraise {
    _ => fail("overflowed audio rows admitted")
  }
}
