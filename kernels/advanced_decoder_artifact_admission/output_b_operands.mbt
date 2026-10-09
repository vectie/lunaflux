///|
fn output_b_projection_operands_are_exact(
  rank : Int,
  groups : Int,
  hidden : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  if rank <= 0 ||
    groups <= 0 ||
    hidden <= 0 ||
    hidden != geometry.hidden_size() ||
    rank > 65536 / groups ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != (hidden + 255) / 256 ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 256 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let rows = dimensions.grid_x().to_int64()
  let input_width = (rank * groups).to_int64()
  let output_width = hidden.to_int64()
  if input_width % 128L != 0L || output_width % 128L != 0L {
    return false
  }
  if rows > 9223372036854775807L / input_width ||
    rows * input_width > 9223372036854775807L / 2L ||
    output_width > 9223372036854775807L / input_width ||
    rows > 9223372036854775807L / output_width ||
    rows * output_width > 9223372036854775807L / 2L {
    return false
  }
  let input_bytes = rows * input_width * 2L
  let weight_bytes = output_width * input_width
  let scale_bytes = output_width / 128L * (input_width / 128L)
  let output_bytes = rows * output_width * 2L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == input_bytes &&
  operands[1].alignment() == 16L &&
  operands[2].byte_count() == weight_bytes &&
  operands[2].alignment() == 16L &&
  operands[3].byte_count() == scale_bytes &&
  operands[3].alignment() == 16L &&
  operands[4].byte_count() == output_bytes &&
  operands[4].alignment() == 16L
}
