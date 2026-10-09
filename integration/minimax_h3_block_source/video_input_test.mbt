///|
test "video output bridge is the inverse packing followed by checkpoint affine" {
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
      inference_steps=2,
    )
  let plan = @joint_diffusion_loop.PreparedPlan::new(
    model,
    request,
    max_latent_elements=100000000L,
  )
  let source = @minimax_h3_block_source.build_video_input(plan)
  assert_eq(source.value_bytes, plan.video_elements() * 4L)
  assert_eq(source.elements_per_channel, 37L * 48L * 84L)
  for index in [0L, 1L, 83L, 84L, 24L * 37L * 48L * 84L - 1L] {
    assert_eq(
      plan.video_pack().source_index(source.permutation.source_index(index)),
      index,
    )
  }
  let text = @utf8.decode(source.source)
  for symbol in source.symbols {
    assert_true(text.contains("void " + symbol + "("))
  }
}
