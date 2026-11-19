///|
async test "recurrent branch binds projections shared reset and packed QKV views" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  for
    (site, shape) in [
      (@operators.LinearProjection::QueryKeyValue, (32, 8192, 2048)),
      (Gate, (32, 4096, 2048)),
      (Decay, (32, 32, 2048)),
      (Beta, (32, 32, 2048)),
      (Output, (32, 2048, 4096)),
    ] {
    assert_eq(operators.linear_projection(site).geometry(), shape)
  }
  let program = @minicpm_v47_recurrent.Program::new(operators)
  let stages = program.stages()
  assert_eq(stages.length(), 9)
  assert_eq(program.regions().length(), 28)
  assert_eq(program.workspace_bytes(), 1845248L)
  assert_eq(program.region(ConvolutionState).byte_count, 49152L)
  assert_eq(program.region(RecurrentState).byte_count, 2097152L)
  assert_eq(program.region(RecurrentState).workspace_offset, None)
  assert_eq(stages[4].arguments(), [
    Counts,
    RowOffsets,
    RequestSlots,
    Reset,
    RawQkv,
    ConvolutionWeight,
    ConvolutionState,
    ConvolvedQuery,
  ])
  assert_eq(stages[5].arguments(), [
    Counts,
    Decay,
    Beta,
    ALog,
    DtBias,
    LogDecay,
    BetaControl,
  ])
  assert_eq(stages[6].arguments(), [
    Counts,
    RowOffsets,
    RequestSlots,
    Reset,
    ConvolvedQuery,
    ConvolvedKey,
    ConvolvedValue,
    LogDecay,
    BetaControl,
    RecurrentState,
    DeltaOutput,
  ])
  assert_eq(stages[7].arguments(), [
    Counts,
    DeltaOutput,
    NormWeight,
    Gate,
    GatedOutput,
  ])
  assert_eq(stages[8].arguments(), [Counts, GatedOutput, OutputWeight, Output])
  let convolved = program.region(ConvolvedQuery)
  let key = program.region(ConvolvedKey)
  let value = program.region(ConvolvedValue)
  assert_eq(
    key.workspace_offset.unwrap() - convolved.workspace_offset.unwrap(),
    4096L,
  )
  assert_eq(
    value.workspace_offset.unwrap() - convolved.workspace_offset.unwrap(),
    8192L,
  )
  assert_true(
    key.workspace_offset.unwrap() + key.byte_count <=
    convolved.workspace_offset.unwrap() + convolved.byte_count,
  )
  assert_eq(
    value.workspace_offset.unwrap() + value.byte_count,
    convolved.workspace_offset.unwrap() + convolved.byte_count,
  )
  let mut end = 0L
  for region in program.regions() {
    if region.slot != ConvolvedKey &&
      region.slot != ConvolvedValue &&
      region.workspace_offset is Some(offset) {
      assert_eq(offset % 16L, 0L)
      assert_true(offset >= end)
      end = offset + region.byte_count
    }
  }
  assert_eq(end, program.workspace_bytes())
  let source = @utf8.decode(program.source_bytes()[:])
  for stage in stages {
    assert_true(source.contains("void " + stage.symbol() + "("))
    for slot in stage.arguments() {
      assert_true(program.region(slot).byte_count > 0L)
    }
  }
  let frame = @minicpm_v47_recurrent.Frame::new(program)
  frame.close()
  frame.close()
}

///|
async test "recurrent state budget follows slot capacity independently of live sequences" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(
    spec,
    maximum_rows=8,
    maximum_sequences=2,
    cache_slots=4,
  )
  let program = @minicpm_v47_recurrent.Program::new(operators)
  assert_eq(program.region(RowOffsets).byte_count, 12L)
  assert_eq(program.region(RequestSlots).byte_count, 8L)
  assert_eq(program.region(Reset).byte_count, 8L)
  assert_eq(program.region(ConvolutionState).byte_count, 196608L)
  assert_eq(program.region(RecurrentState).byte_count, 8388608L)
  assert_eq(program.workspace_bytes(), 461312L)
  assert_eq(program.stages()[4].launch(), (64, 2, 128))
  assert_eq(program.stages()[6].launch(), (2, 32, 128))
}
