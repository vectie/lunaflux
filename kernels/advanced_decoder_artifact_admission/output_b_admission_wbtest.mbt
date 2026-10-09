///|
test "Output-B artifact ABI admits exact raw code and scale operands only" {
  let requirements = output_a_admission_requirements(
    hidden_size=128,
    output_rank=128,
  )
  let mut ordinal = -1
  for index, requirement in requirements.requirements() {
    if requirement.operation() is AttentionOutputBProjectionV1(_, _, _) {
      ordinal = index
    }
  }
  let operation = requirements.requirements()[ordinal].operation()
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let exact = [
    try! AdvancedDecoderLaunchOperand::new(
      role=StepCountsInput,
      byte_count=20L,
      alignment=4L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=AttentionOutputRankInput,
      byte_count=512L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=AttentionOutputBWeightCodesInput,
      byte_count=16384L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=AttentionOutputBWeightScalesInput,
      byte_count=1L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=HiddenOutput,
      byte_count=512L,
      alignment=16L,
    ),
  ]
  assert_true(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(exact),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong = exact.copy()
  wrong[3] = try! AdvancedDecoderLaunchOperand::new(
    role=AttentionOutputBWeightCodesInput,
    byte_count=1L,
    alignment=16L,
  )
  assert_false(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(wrong),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let short = exact.copy()
  short[2] = try! AdvancedDecoderLaunchOperand::new(
    role=AttentionOutputBWeightCodesInput,
    byte_count=16383L,
    alignment=16L,
  )
  assert_false(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(short),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
}
