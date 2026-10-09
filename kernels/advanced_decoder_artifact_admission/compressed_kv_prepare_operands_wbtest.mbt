///|
fn compressed_kv_operand(
  role : AdvancedDecoderLaunchOperandRole,
  bytes : Int64,
  alignment? : Int64 = 2L,
) -> AdvancedDecoderLaunchOperand {
  try! AdvancedDecoderLaunchOperand::new(role~, byte_count=bytes, alignment~)
}

///|
test "compressed KV prepare admission pins direction bytes and launch" {
  let geometry = deterministic_index_test_requirements().geometry()
  let dimensions = try! @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
  let operands = [
    compressed_kv_operand(StepCountsInput, 20L, alignment=4L),
    compressed_kv_operand(PreparedKeyValueModesInput, 4L, alignment=4L),
    compressed_kv_operand(
      PreparedKeyValueSequenceLengthsInput,
      4L,
      alignment=4L,
    ),
    compressed_kv_operand(UncompressedKeyValueInput, 131072L),
    compressed_kv_operand(CompressedKeyValueInput, 1024L),
    compressed_kv_operand(PreparedKeyValueLengthsOutput, 4L, alignment=4L),
    compressed_kv_operand(PreparedKeyValueOutput, 132096L),
  ]
  assert_true(
    compressed_kv_prepare_operands_are_exact(
      64,
      128,
      ReadOnlyArray::from_array(operands),
      dimensions,
      geometry,
    ),
  )
  let wrong_role = operands.copy()
  wrong_role[3] = compressed_kv_operand(CompressedKeyValueInput, 131072L)
  assert_false(
    compressed_kv_prepare_operands_are_exact(
      64,
      128,
      ReadOnlyArray::from_array(wrong_role),
      dimensions,
      geometry,
    ),
  )
  let wrong_bytes = operands.copy()
  wrong_bytes[6] = compressed_kv_operand(PreparedKeyValueOutput, 132094L)
  assert_false(
    compressed_kv_prepare_operands_are_exact(
      64,
      128,
      ReadOnlyArray::from_array(wrong_bytes),
      dimensions,
      geometry,
    ),
  )
}
