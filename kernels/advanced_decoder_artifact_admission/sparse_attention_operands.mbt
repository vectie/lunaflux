///|
fn selected_sparse_attention_operands_are_exact(
  window_size : Int,
  compression_ratio : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let maximum_positions = geometry.maximum_position_count()
  let heads = geometry.attention_head_count()
  let dimension = geometry.attention_head_dimension()
  if window_size <= 0 ||
    compression_ratio <= 1 ||
    maximum_positions <= 0 ||
    window_size > maximum_positions ||
    compression_ratio > maximum_positions ||
    heads <= 0 ||
    dimension <= 0 ||
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
  let head_count = heads.to_int64()
  let head_width = dimension.to_int64()
  let selected_width = (window_size + maximum_positions / compression_ratio).to_int64()
  let maximum_key_slots = (maximum_positions +
  maximum_positions / compression_ratio).to_int64()
  if operands[4].byte_count() % 4L != 0L {
    return false
  }
  let maximum_batches = operands[4].byte_count() / 4L
  if maximum_batches <= 0L ||
    maximum_batches > rows ||
    rows > 9223372036854775807L / 4L ||
    rows > 9223372036854775807L / head_count ||
    rows * head_count > 9223372036854775807L / head_width ||
    rows * head_count * head_width > 9223372036854775807L / 2L ||
    maximum_batches > 9223372036854775807L / maximum_key_slots ||
    maximum_batches * maximum_key_slots > 9223372036854775807L / head_width ||
    maximum_batches * maximum_key_slots * head_width > 9223372036854775807L / 2L ||
    rows > 9223372036854775807L / selected_width ||
    rows * selected_width > 9223372036854775807L / 4L {
    return false
  }
  let query_bytes = rows * head_count * head_width * 2L
  let key_value_bytes = maximum_batches * maximum_key_slots * head_width * 2L
  let selected_bytes = rows * selected_width * 4L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == query_bytes &&
  operands[1].alignment() == 16L &&
  operands[2].byte_count() == rows * 4L &&
  operands[2].alignment() == 4L &&
  operands[3].byte_count() == key_value_bytes &&
  operands[3].alignment() == 16L &&
  operands[4].alignment() == 4L &&
  operands[5].byte_count() == head_count * 4L &&
  operands[5].alignment() == 4L &&
  operands[6].byte_count() == selected_bytes &&
  operands[6].alignment() == 4L &&
  operands[7].byte_count() == query_bytes &&
  operands[7].alignment() == 16L
}
