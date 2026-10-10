///|
test "shared prediction geometry preserves every existing exported source and symbol" {
  let spec = @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
    variant=Fl2va,
    checkpoint_format=DiffusersV1,
  )
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      spec,
      @spec.ContentDigest::from_sha256(String::make(64, '9')),
    ),
  )
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=8,
    audio_rows=4,
    conditioning=[],
    max_rows=64,
  )
  for count in [1, 2, 3, 4] {
    let sources = @minimax_h3_block_source.build_prediction_sources(
      model,
      sequence,
      distinct_timesteps=count,
      block_norm_epsilon=spec.norm_epsilon(),
      function_prefix="h3",
    )
    let block = @minimax_h3_block_source.build(
      model,
      sequence,
      distinct_timesteps=count,
      block_norm_epsilon=spec.norm_epsilon(),
      function_prefix="h3_block_n\{count}",
    )
    let time = @minimax_h3_block_source.build_timestep(
      model,
      distinct_timesteps=count,
      function_prefix="h3_timestep_n\{count}",
    )
    let output = @minimax_h3_block_source.build_output(
      model,
      sequence,
      distinct_timesteps=count,
      function_prefix="h3_output_n\{count}",
    )
    assert_eq(sources.block.source, block.source)
    assert_eq(sources.block.symbols, block.symbols)
    assert_eq(sources.timestep.source, time.source)
    assert_eq(sources.timestep.symbols, time.symbols)
    assert_eq(sources.output.source, output.source)
    assert_eq(sources.output.symbols, output.symbols)
    assert_eq(sources.block.packed_mapping, sources.output.packed_mapping)
  }
  for count in [0, 5] {
    try
      @minimax_h3_block_source.build_prediction_sources(
        model,
        sequence,
        distinct_timesteps=count,
        block_norm_epsilon=spec.norm_epsilon(),
        function_prefix="h3",
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("unsupported schedule cardinality accepted")
    }
  }
}
