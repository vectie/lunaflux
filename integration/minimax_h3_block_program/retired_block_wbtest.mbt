///|
test "unprepared block observation cannot read a retired or absent workspace" {
  let model = @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant=Fl2va,
        checkpoint_format=DiffusersV1,
      ),
      @spec.ContentDigest::from_sha256(String::make(64, '7')),
    ),
  )
  let sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=592,
    audio_rows=414,
    conditioning=[],
    max_rows=4096,
  )
  let source = @minimax_h3_block_source.build_prediction_sources(
    model,
    sequence,
    distinct_timesteps=1,
    block_norm_epsilon=0.000001,
    function_prefix="observation",
  )
  let variant = PredictionVariant::new(
    Program::new(source.block),
    TimestepProgram::new(source.timestep),
    OutputProgram::new(source.output),
  )
  try variant.capture_block_stage(Attention) catch {
    @device.DeviceError::InvalidArgument(CopyToHost) => ()
    _ => fail("unexpected observation error")
  } noraise {
    _ => fail("unprepared workspace observed")
  }
}
