///|
fn deterministic_compressed_index_operands_are_exact(
  compression_ratio : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let maximum_positions = geometry.maximum_position_count()
  if compression_ratio <= 1 ||
    maximum_positions <= 0 ||
    compression_ratio > maximum_positions ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != 1 ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 256 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let rows = dimensions.grid_x().to_int64()
  let slots = (maximum_positions / compression_ratio).to_int64()
  if slots <= 0L ||
    rows > 9223372036854775807L / 4L ||
    rows > 9223372036854775807L / slots ||
    rows * slots > 9223372036854775807L / 4L {
    return false
  }
  let row_bytes = rows * 4L
  let index_bytes = rows * slots * 4L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == row_bytes &&
  operands[1].alignment() == 4L &&
  operands[2].byte_count() == row_bytes &&
  operands[2].alignment() == 4L &&
  operands[3].byte_count() == row_bytes &&
  operands[3].alignment() == 4L &&
  operands[4].byte_count() == index_bytes &&
  operands[4].alignment() == 4L
}
