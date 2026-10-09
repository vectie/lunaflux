///|
fn head_reduce_operands_are_exact(
  width : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let hidden = geometry.hidden_size()
  if width <= 1 ||
    width > 16 ||
    hidden <= 0 ||
    hidden > 65536 ||
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
  let stream = width.to_int64()
  let feature = hidden.to_int64()
  if stream > 9223372036854775807L / feature ||
    rows > 9223372036854775807L / stream ||
    rows * stream > 9223372036854775807L / feature ||
    rows * stream * feature > 9223372036854775807L / 2L ||
    stream * feature > 9223372036854775807L / stream ||
    stream * stream * feature > 9223372036854775807L / 4L ||
    rows > 9223372036854775807L / feature ||
    rows * feature > 9223372036854775807L / 2L {
    return false
  }
  let state_bytes = rows * stream * feature * 2L
  let function_bytes = stream * stream * feature * 4L
  let base_bytes = stream * 4L
  let output_bytes = rows * feature * 2L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == state_bytes &&
  operands[1].alignment() == 16L &&
  operands[2].byte_count() == function_bytes &&
  operands[2].alignment() == 16L &&
  operands[3].byte_count() == base_bytes &&
  operands[3].alignment() == 16L &&
  operands[4].byte_count() == 4L &&
  operands[4].alignment() == 4L &&
  operands[5].byte_count() == output_bytes &&
  operands[5].alignment() == 16L
}
