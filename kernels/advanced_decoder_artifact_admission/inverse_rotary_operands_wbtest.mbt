///|
fn inverse_rotary_operand(
  role : AdvancedDecoderLaunchOperandRole,
  bytes : Int64,
  alignment : Int64,
) -> AdvancedDecoderLaunchOperand {
  try! AdvancedDecoderLaunchOperand::new(role~, byte_count=bytes, alignment~)
}

///|
test "inverse RoPE admission pins full-head directional ABI" {
  let geometry = deterministic_index_test_requirements().geometry()
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=2,
    grid_y=2,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let operands = [
    inverse_rotary_operand(StepCountsInput, 20L, 4L),
    inverse_rotary_operand(PositionInput, 8L, 4L),
    inverse_rotary_operand(AttentionOutputRotaryInOut, 512L, 16L),
  ]
  assert_true(
    attention_output_inverse_rotary_operands_are_exact(
      128,
      ReadOnlyArray::from_array(operands),
      dimensions,
      geometry,
    ),
  )
  let wrong_role = operands.copy()
  wrong_role[2] = inverse_rotary_operand(AttentionOutputRotaryInOut, 512L, 4L)
  assert_false(
    attention_output_inverse_rotary_operands_are_exact(
      128,
      ReadOnlyArray::from_array(wrong_role),
      dimensions,
      geometry,
    ),
  )
  let wrong_bytes = operands.copy()
  wrong_bytes[2] = inverse_rotary_operand(AttentionOutputRotaryInOut, 510L, 16L)
  assert_false(
    attention_output_inverse_rotary_operands_are_exact(
      128,
      ReadOnlyArray::from_array(wrong_bytes),
      dimensions,
      geometry,
    ),
  )
}
