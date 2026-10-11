///|
test "DiT probability intervention changes only its declared storage boundary" {
  let config = @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
    variant=Fl2va,
    checkpoint_format=DiffusersV1,
  )
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      config,
      @spec.ContentDigest::from_sha256(String::make(64, '7')),
    ),
  )
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=2368,
    audio_rows=414,
    conditioning=[],
    max_rows=4096,
  )
  let baseline = @minimax_h3_block_source.build_prediction_sources(
    model,
    sequence,
    distinct_timesteps=1,
    block_norm_epsilon=config.norm_epsilon(),
    function_prefix="h3",
  )
  let candidate = @minimax_h3_block_source.build_prediction_sources(
    model,
    sequence,
    distinct_timesteps=1,
    block_norm_epsilon=config.norm_epsilon(),
    function_prefix="h3",
    probability_law=GlobalUnnormalizedBf16,
  )
  assert_eq(baseline.block.rows, 2816)
  assert_eq(baseline.block.symbols, candidate.block.symbols)
  assert_eq(baseline.timestep.source, candidate.timestep.source)
  assert_eq(baseline.output.source, candidate.output.source)
  let source = @utf8.decode(candidate.block.source)
  assert_eq(
    source.replace(
      old="__bfloat162float(__float2bfloat16_rn(probability))",
      new="probability",
    ),
    @utf8.decode(baseline.block.source),
  )
  if @env.get_env_var("LUNA_EXPORT_BLOCK_PROBABILITY") == Some("1") {
    println("LUNA_BLOCK_BASELINE_BEGIN")
    println(@utf8.decode(baseline.block.source))
    println("LUNA_BLOCK_BASELINE_END")
    println("LUNA_BLOCK_CANDIDATE_BEGIN")
    println(source)
    println("LUNA_BLOCK_CANDIDATE_END")
  }
}
