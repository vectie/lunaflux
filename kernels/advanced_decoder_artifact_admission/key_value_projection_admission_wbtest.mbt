///|
test "key/value projection artifact ABI is directional and exact" {
  let requirements = output_a_admission_requirements(
    hidden_size=128,
    query_rank=128,
    attention_head_dimension=128,
  )
  let operation = requirements.requirements()[2].operation()
  assert_eq(operation, KeyValueProjection(1, 128, 64))
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
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
      byte_count=256L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=KeyValueWeightCodesInput,
      byte_count=16384L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=KeyValueWeightScalesInput,
      byte_count=1L,
      alignment=16L,
    ),
    try! AdvancedDecoderLaunchOperand::new(
      role=KeyValueProjectionOutput,
      byte_count=256L,
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
  let wrong_role = exact.copy()
  wrong_role[4] = try! AdvancedDecoderLaunchOperand::new(
    role=KeyValueOutput,
    byte_count=256L,
    alignment=16L,
  )
  assert_false(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(wrong_role),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
  let wrong_scale_bytes = exact.copy()
  wrong_scale_bytes[3] = try! AdvancedDecoderLaunchOperand::new(
    role=KeyValueWeightScalesInput,
    byte_count=2L,
    alignment=16L,
  )
  assert_false(
    operands_are_exact(
      operation,
      ReadOnlyArray::from_array(wrong_scale_bytes),
      dimensions,
      requirements.geometry(),
      8,
    ),
  )
}
