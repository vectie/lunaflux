///|
fn window_compressed_join_operands_are_exact(
  window_size : Int,
  compression_ratio : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let maximum_positions = geometry.maximum_position_count()
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    maximum_positions <= 0 ||
    window_size > maximum_positions ||
    compression_ratio > maximum_positions ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != 1 ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 1 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let rows = dimensions.grid_x().to_int64()
  let compressed_width = (maximum_positions / compression_ratio).to_int64()
  let joined_width = window_size.to_int64() + compressed_width
  if rows > 9223372036854775807L / 4L ||
    rows > 9223372036854775807L / compressed_width ||
    rows * compressed_width > 9223372036854775807L / 4L ||
    rows > 9223372036854775807L / joined_width ||
    rows * joined_width > 9223372036854775807L / 4L {
    return false
  }
  let row_bytes = rows * 4L
  let compressed_bytes = rows * compressed_width * 4L
  let joined_bytes = rows * joined_width * 4L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == row_bytes &&
  operands[1].alignment() == 4L &&
  operands[2].byte_count() == row_bytes &&
  operands[2].alignment() == 4L &&
  operands[3].byte_count() == row_bytes &&
  operands[3].alignment() == 4L &&
  operands[4].byte_count() == row_bytes &&
  operands[4].alignment() == 4L &&
  operands[5].byte_count() == compressed_bytes &&
  operands[5].alignment() == 4L &&
  operands[6].byte_count() == row_bytes &&
  operands[6].alignment() == 4L &&
  operands[7].byte_count() == joined_bytes &&
  operands[7].alignment() == 4L
}
