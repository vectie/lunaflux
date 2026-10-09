///|
test "complete block bundle shares geometry and isolates padding" {
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
  let time = @minimax_h3_block_source.build_timestep(
    model,
    distinct_timesteps=3,
    function_prefix="h3_time",
  )
  assert_eq(time.symbols.length(), 4)
  assert_eq(time.output_width, 2688)
  let time_source = @utf8.decode(time.source)
  let refiner = @minimax_h3_block_source.build_refiner(
    model,
    text_rows=3,
    block_norm_epsilon=config.norm_epsilon(),
    function_prefix="h3_refiner",
  )
  assert_eq(refiner.symbols.length(), 10)
  assert_eq(refiner.symbols[9], "h3_refiner_context")
  let refiner_source = @utf8.decode(refiner.source)
  for symbol in refiner.symbols {
    assert_true(refiner_source.contains("void " + symbol + "("))
  }
  for symbol in time.symbols {
    assert_true(time_source.contains("void " + symbol + "("))
  }
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=8,
    audio_rows=4,
    conditioning=[],
    max_rows=64,
  )
  let reference_sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=8,
    audio_rows=4,
    conditioning=[Image(2), Video(audio_rows=1, video_rows=4), Audio(0)],
    max_rows=64,
  )
  let conditions = @minimax_h3_block_source.build_conditioning(
    model,
    reference_sequence,
    function_prefix="h3_condition",
  )
  assert_eq(conditions.copies.length(), 4)
  assert_eq(conditions.copies[0].rows, 3)
  assert_eq(conditions.copies[1].rows, 2)
  assert_eq(conditions.copies[2].rows, 1)
  assert_eq(conditions.copies[3].rows, 4)
  assert_eq(conditions.copies[3].destination_offset, 6L * 5376L * 2L)
  assert_eq(conditions.byte_count, 10L * 5376L * 2L)
  let condition_source = @utf8.decode(conditions.source)
  for symbol in conditions.symbols {
    assert_true(condition_source.contains("void " + symbol + "("))
  }
  assert_true(condition_source.contains("output[i] = input[i]"))
  assert_eq(conditions.symbols.length(), 7)
  assert_eq(conditions.copies[1].input_width, 96)
  assert_eq(conditions.copies[2].input_width, 32)
  assert_eq(conditions.copies[3].input_width, 96)
  let bundle = @minimax_h3_block_source.build(
    model,
    sequence,
    distinct_timesteps=2,
    block_norm_epsilon=config.norm_epsilon(),
    function_prefix="h3_block",
  )
  let input = @minimax_h3_block_source.build_input(
    model,
    sequence,
    function_prefix="h3_input",
  )
  let input_source = @utf8.decode(input.source)
  assert_eq(input.symbols.length(), 3)
  for symbol in input.symbols {
    assert_true(input_source.contains("void " + symbol + "("))
  }
  assert_eq(bundle.rows, 64)
  assert_eq(bundle.distinct_timesteps, 2)
  assert_eq(bundle.symbols.length(), 11)
  let source = @utf8.decode(bundle.source)
  for symbol in bundle.symbols {
    assert_true(source.contains("void " + symbol + "("))
  }
  assert_true(source.contains("if (qr >= 15ULL) { key_begin = 15;"))
  let final_output = @minimax_h3_block_source.build_output(
    model,
    sequence,
    distinct_timesteps=2,
    function_prefix="h3_final",
  )
  assert_eq(final_output.video_start, 7)
  assert_eq(final_output.video_rows, 8)
  assert_eq(final_output.audio_start, 3)
  assert_eq(final_output.audio_rows, 4)
  let final_source = @utf8.decode(final_output.source)
  assert_eq(final_output.symbols.length(), 6)
  for symbol in final_output.symbols {
    assert_true(final_source.contains("void " + symbol + "("))
  }
  try
    @minimax_h3_block_source.build(
      model,
      sequence,
      distinct_timesteps=0,
      block_norm_epsilon=config.norm_epsilon(),
      function_prefix="h3_block",
    )
  catch {
    @minimax_h3_block_source.BlockSourceError::InvalidShape => ()
    _ => fail("unexpected error")
  } noraise {
    _ => fail("accepted empty timestep table")
  }
}
