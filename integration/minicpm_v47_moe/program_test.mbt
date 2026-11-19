///|
async test "prepared MoE stage ABIs match its nine generated entrypoints" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  let program = @minicpm_v47_moe.Program::new(operators)
  let stages = program.stages()
  assert_eq(stages.length(), 9)
  let expected : Array[Array[@minicpm_v47_moe.Slot]] = [
    [Counts, Input, RouterWeight, RouterLogits],
    [Counts, RouterLogits, SelectedIndices, SelectedWeights],
    [Counts, Input, SelectedIndices, ExpertMap, RoutedBank, ExpertActivation],
    [
      Counts,
      ExpertActivation,
      SelectedIndices,
      SelectedWeights,
      ExpertMap,
      RoutedBank,
      ExpertWeighted,
    ],
    [Counts, SelectedIndices, ExpertWeighted, RoutedOutput],
    [Counts, Input, SharedGateWeight, SharedUpWeight, SharedProduct],
    [Counts, SharedProduct, SharedDownWeight, SharedOutput],
    [Counts, Input, SharedGateProjectionWeight, SharedGateLogits],
    [Counts, RoutedOutput, SharedOutput, SharedGateLogits, Output],
  ]
  let source = @utf8.decode(program.source_bytes()[:])
  for i, stage in stages {
    assert_eq(stage.arguments(), ReadOnlyArray::from_array(expected[i]))
    assert_true(source.contains("void " + stage.symbol() + "("))
  }
  assert_eq(stages[0].launch(), (64, 1, 128))
  assert_eq(stages[3].launch(), (131072, 1, 128))
  assert_eq(stages[8].launch(), (32, 1, 128))
  assert_eq(program.region(SelectedWeights).byte_count, 1024L)
  assert_eq(program.region(RoutedOutput).byte_count, 262144L)
  assert_eq(program.region(RoutedBank).byte_count, 1610612736L)
}

///|
async test "one aligned MoE workspace excludes borrowed weights and can be shared across layers" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let program = @minicpm_v47_moe.Program::new(
    @operators.DecoderOperators::new(spec, maximum_rows=32),
  )
  let mut end = 0L
  let mut scratch_count = 0
  for region in program.regions() {
    if region.workspace_offset is Some(offset) {
      assert_eq(offset % 16L, 0L)
      assert_eq(offset % region.alignment, 0L)
      assert_true(offset >= end)
      end = offset + region.byte_count
      scratch_count += 1
    }
  }
  assert_eq(scratch_count, 10)
  assert_eq(end, program.workspace_bytes())
  assert_eq(program.workspace_bytes(), 1756224L)
  assert_true(program.region(RoutedBank).workspace_offset is None)
  assert_true(program.region(Input).workspace_offset is None)
  assert_eq(program.region(ExpertMap).byte_count, 1024L)
  // Closing a fresh frame needs no device. Actual handle/lifecycle qualification
  // remains a target-device test, not simulated by this check.
  let frame = @minicpm_v47_moe.Frame::new(program)
  frame.close()
  frame.close()
  assert_true(
    try frame.launches() catch {
      _ => true
    } noraise {
      _ => false
    },
  )
}
