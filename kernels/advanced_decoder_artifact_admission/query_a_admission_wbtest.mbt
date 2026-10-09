///|
test "Query-A artifact ABI admits exact replicated code and scale roles" {
  let requirements = output_a_admission_requirements(
    hidden_size=128,
    query_rank=128,
    output_rank=128,
  )
  let operation = requirements.requirements()[1].operation()
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
      role=HiddenInput,
      byte_count=512L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=QueryAWeightCodesInput,
      byte_count=16384L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=QueryAWeightScalesInput,
      byte_count=1L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=QueryLowRankAOutput,
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
  wrong[4] = try! AdvancedDecoderLaunchOperand::new(
    role=QueryOutput,
    byte_count=512L,
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
    role=QueryAWeightCodesInput,
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
