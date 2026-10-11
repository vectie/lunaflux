///|
test "offline prefix observer borrows no unprepared timestep storage" {
  let source = prediction_program_fixture()[0]
  let variant = PredictionVariant::new(
    Program::new(source.block),
    TimestepProgram::new(source.timestep),
    OutputProgram::new(source.output),
  )
  let input = InputProgram::new(
    @minimax_h3_block_source.build_input(
      source.block.model,
      @minimax_h3_sequence.Sequence::new(
        hidden_width=5376,
        text_rows=3,
        video_rows=8,
        audio_rows=4,
        conditioning=[],
        max_rows=64,
      ),
      function_prefix="h3_input",
    ),
  )
  assert_eq(
    variant.input_launch_count(input),
    source.timestep.symbols.length() + 3,
  )
  let observed = Ref(false)
  try variant.with_timestep_embedding((_, _, _) => observed.val = true) catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorPoll) => ()
    _ => fail("wrong observer lifecycle error")
  } noraise {
    _ => fail("unprepared timestep exposed")
  }
  assert_false(observed.val)
  input.close()
  variant.block.close()
  variant.timestep.close()
  variant.output.close()
}
