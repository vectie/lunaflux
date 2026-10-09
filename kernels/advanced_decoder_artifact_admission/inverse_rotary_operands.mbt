///|
fn attention_output_inverse_rotary_operands_are_exact(
  original_positions : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  let heads = geometry.attention_head_count()
  let head_dimension = geometry.attention_head_dimension()
  let rope_dimension = geometry.rope_head_dimension()
  if operands.length() != 3 ||
    operands[0].role() != StepCountsInput ||
    operands[1].role() != PositionInput ||
    operands[2].role() != AttentionOutputRotaryInOut ||
    original_positions <= 0 ||
    original_positions > geometry.maximum_position_count() ||
    heads <= 0 ||
    head_dimension <= 0 ||
    head_dimension % 2 != 0 ||
    rope_dimension != 64 ||
    rope_dimension > head_dimension ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != heads ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != head_dimension / 2 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let rows = dimensions.grid_x().to_int64()
  let head_count = heads.to_int64()
  let width = head_dimension.to_int64()
  if rows > 9223372036854775807L / 4L ||
    rows > 9223372036854775807L / head_count ||
    rows * head_count > 9223372036854775807L / width ||
    rows * head_count * width > 9223372036854775807L / 2L {
    return false
  }
  let position_bytes = rows * 4L
  let output_bytes = rows * head_count * width * 2L
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == position_bytes &&
  operands[1].alignment() == 4L &&
  operands[2].byte_count() == output_bytes &&
  operands[2].alignment() == 16L
}
