///|
fn prediction_program_fixture() -> Array[
  @minimax_h3_block_source.PredictionSources,
] raise {
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
  [1, 2, 4].map(count => {
    @minimax_h3_block_source.build_prediction_sources(
      model,
      sequence,
      distinct_timesteps=count,
      block_norm_epsilon=spec.norm_epsilon(),
      function_prefix="h3",
    )
  })
}

///|
test "prediction programs preflight the sum of all actual variant workspaces" {
  let sources = prediction_program_fixture()
  let (layouts, total) = prediction_workspaces(sources[:], 1073741824L)
  assert_eq(layouts.length(), 3)
  assert_true(total > layouts[0].block)
  assert_eq(
    total,
    layouts.fold(init=0L, (sum, p) => sum + p.block + p.timestep + p.output),
  )
  let owner = PredictionPrograms::new(sources[:], max_workspace_bytes=total)
  assert_eq(owner.workspace_bytes(), total)
  assert_true(owner.programs.is_empty())
  try
    PredictionPrograms::new(sources[:], max_workspace_bytes=total - 1L)
  catch {
    @device.DeviceError::InvalidArgument(AllocationCreate) => ()
    _ => fail("wrong aggregate budget failure")
  } noraise {
    _ => fail("aggregate budget was treated as a per-variant budget")
  }
  try owner.variants() catch {
    @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
    _ => fail("wrong unprepared variant failure")
  } noraise {
    _ => fail("unprepared programs exposed")
  }
  owner.close()
  owner.close()
  assert_true(owner.started && !owner.ready)
}

///|
test "prediction program variants reject duplicate or cross-request geometry before allocation" {
  let sources = prediction_program_fixture()
  for wrong in [[], [sources[0], sources[0]]] {
    try
      PredictionPrograms::new(wrong[:], max_workspace_bytes=1073741824L)
    catch {
      _ => ()
    } noraise {
      _ => fail("empty or duplicate variants accepted")
    }
  }
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
  let other_sequence = @minimax_h3_sequence.Sequence::new(
    hidden_width=model.denoiser().hidden_size(),
    text_rows=3,
    video_rows=9,
    audio_rows=4,
    conditioning=[],
    max_rows=64,
  )
  let other = @minimax_h3_block_source.build_prediction_sources(
    model,
    other_sequence,
    distinct_timesteps=2,
    block_norm_epsilon=spec.norm_epsilon(),
    function_prefix="h3",
  )
  try
    PredictionPrograms::new(
      [sources[0], other][:],
      max_workspace_bytes=1073741824L,
    )
  catch {
    @device.DeviceError::InvalidArgument(OrderedExecutorCreate) => ()
    _ => fail("wrong geometry failure")
  } noraise {
    _ => fail("same row capacity with different row mapping accepted")
  }
}
