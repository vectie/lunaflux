///|
fn query_key_norm_operands_are_exact(
  query_width : Int,
  key_value_width : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Bool {
  if query_width <= 0 ||
    key_value_width <= 0 ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != 2 ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 256 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 1024 {
    return false
  }
  let rows = dimensions.grid_x().to_int64()
  let query_row_bytes = query_width.to_int64() * 2L
  let key_value_row_bytes = key_value_width.to_int64() * 2L
  if query_row_bytes <= 0L ||
    key_value_row_bytes <= 0L ||
    rows > 9223372036854775807L / query_row_bytes ||
    rows > 9223372036854775807L / key_value_row_bytes {
    return false
  }
  let query_bytes = rows * query_row_bytes
  let key_value_bytes = rows * key_value_row_bytes
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == query_bytes &&
  operands[1].alignment() == 16L &&
  operands[2].byte_count() == query_row_bytes &&
  operands[2].alignment() == 16L &&
  operands[3].byte_count() == query_bytes &&
  operands[3].alignment() == 16L &&
  operands[4].byte_count() == key_value_bytes &&
  operands[4].alignment() == 16L &&
  operands[5].byte_count() == key_value_row_bytes &&
  operands[5].alignment() == 16L &&
  operands[6].byte_count() == key_value_bytes &&
  operands[6].alignment() == 16L
}
