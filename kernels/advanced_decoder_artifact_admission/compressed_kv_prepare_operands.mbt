///|
fn compressed_kv_prepare_operands_are_exact(
  window_size : Int,
  compression_ratio : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let maximum_positions = geometry.maximum_position_count()
  let head_dimension = geometry.attention_head_dimension()
  if operands.length() != 7 ||
    operands[0].role() != StepCountsInput ||
    operands[1].role() != PreparedKeyValueModesInput ||
    operands[2].role() != PreparedKeyValueSequenceLengthsInput ||
    operands[3].role() != UncompressedKeyValueInput ||
    operands[4].role() != CompressedKeyValueInput ||
    operands[5].role() != PreparedKeyValueLengthsOutput ||
    operands[6].role() != PreparedKeyValueOutput ||
    window_size <= 0 ||
    compression_ratio <= 1 ||
    window_size > maximum_positions ||
    compression_ratio > maximum_positions ||
    head_dimension <= 0 ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != 1 ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 1 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let batches = dimensions.grid_x().to_int64()
  let positions = maximum_positions.to_int64()
  let compressed = (maximum_positions / compression_ratio).to_int64()
  let head = head_dimension.to_int64()
  if batches > 9223372036854775807L / 4L ||
    positions > 9223372036854775807L / head ||
    compressed > 9223372036854775807L / head {
    return false
  }
  let ordinary_values = positions * head
  let compressed_values = compressed * head
  let maximum_bf16_values_per_batch = 9223372036854775807L / batches / 2L
  if ordinary_values > maximum_bf16_values_per_batch ||
    compressed_values > maximum_bf16_values_per_batch ||
    ordinary_values > 9223372036854775807L - compressed_values ||
    ordinary_values + compressed_values > maximum_bf16_values_per_batch {
    return false
  }
  let row_bytes = batches * 4L
  let ordinary_bytes = batches * ordinary_values * 2L
  let compressed_bytes = batches * compressed_values * 2L
  let prepared_bytes = batches * (ordinary_values + compressed_values) * 2L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == row_bytes &&
  operands[1].alignment() == 4L &&
  operands[2].byte_count() == row_bytes &&
  operands[2].alignment() == 4L &&
  operands[3].byte_count() == ordinary_bytes &&
  operands[3].alignment() == 2L &&
  operands[4].byte_count() == compressed_bytes &&
  operands[4].alignment() == 2L &&
  operands[5].byte_count() == row_bytes &&
  operands[5].alignment() == 4L &&
  operands[6].byte_count() == prepared_bytes &&
  operands[6].alignment() == 2L
}
