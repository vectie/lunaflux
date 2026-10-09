///|
fn window_join_operand(
  role : AdvancedDecoderLaunchOperandRole,
  bytes : Int64,
) -> AdvancedDecoderLaunchOperand {
  try! AdvancedDecoderLaunchOperand::new(role~, byte_count=bytes, alignment=4L)
}

///|
fn window_join_operands() -> Array[AdvancedDecoderLaunchOperand] {
  [
    window_join_operand(StepCountsInput, 20L),
    window_join_operand(WindowQueryPositionsInput, 8L),
    window_join_operand(WindowSequenceLengthsInput, 8L),
    window_join_operand(WindowQueryModesInput, 8L),
    window_join_operand(CompressedIndexCountsInput, 8L),
    window_join_operand(CompressedIndicesInput, 64L),
    window_join_operand(JoinedIndexCountsOutput, 8L),
    window_join_operand(JoinedIndicesOutput, 576L),
  ]
}

///|
fn window_join_dimensions(
  block_x? : Int = 1,
) -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=1,
    grid_z=1,
    block_x~,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
test "window-compressed join admission binds directional roles and extents" {
  let requirements = deterministic_index_test_requirements()
  assert_true(
    operands_are_exact(
      DeterministicWindowCompressedIndexJoinV1(64, 128),
      ReadOnlyArray::from_array(window_join_operands()),
      window_join_dimensions(),
      requirements.geometry(),
      8,
    ),
  )
}

///|
test "window-compressed join admission rejects role bytes and launch mismatch" {
  let requirements = deterministic_index_test_requirements()
  let wrong_role = window_join_operands()
  wrong_role[3] = window_join_operand(PositionInput, 8L)
  assert_false(
    operands_are_exact(
      DeterministicWindowCompressedIndexJoinV1(64, 128),
      ReadOnlyArray::from_array(wrong_role),
      window_join_dimensions(),
      requirements.geometry(),
      8,
    ),
  )
  let wrong_bytes = window_join_operands()
  wrong_bytes[7] = window_join_operand(JoinedIndicesOutput, 572L)
  assert_false(
    operands_are_exact(
      DeterministicWindowCompressedIndexJoinV1(64, 128),
      ReadOnlyArray::from_array(wrong_bytes),
      window_join_dimensions(),
      requirements.geometry(),
      8,
    ),
  )
  assert_false(
    operands_are_exact(
      DeterministicWindowCompressedIndexJoinV1(64, 128),
      ReadOnlyArray::from_array(window_join_operands()),
      window_join_dimensions(block_x=32),
      requirements.geometry(),
      8,
    ),
  )
}
