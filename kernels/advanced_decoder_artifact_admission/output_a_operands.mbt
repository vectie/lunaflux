///|
fn output_a_projection_operands_are_exact(
  rank : Int,
  groups : Int,
  heads : Int,
  head_dimension : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  geometry : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelGeometry,
) -> Bool {
  if rank <= 0 ||
    groups <= 0 ||
    heads <= 0 ||
    head_dimension <= 0 ||
    heads % groups != 0 ||
    heads != geometry.attention_head_count() ||
    head_dimension != geometry.attention_head_dimension() ||
    dimensions.grid_x() <= 0 ||
    dimensions.grid_y() != groups ||
    dimensions.grid_z() != 1 ||
    dimensions.block_x() != 256 ||
    dimensions.block_y() != 1 ||
    dimensions.block_z() != 1 ||
    dimensions.shared_memory_bytes() != 0 {
    return false
  }
  let rows = dimensions.grid_x().to_int64()
  let head_count = heads.to_int64()
  let width = head_dimension.to_int64()
  let group_count = groups.to_int64()
  let output_rank = rank.to_int64()
  if rows > 9223372036854775807L / head_count ||
    rows * head_count > 9223372036854775807L / width ||
    rows * head_count * width > 9223372036854775807L / 2L ||
    output_rank > 9223372036854775807L / head_count ||
    output_rank * head_count > 9223372036854775807L / width ||
    output_rank * head_count * width > 9223372036854775807L / 2L ||
    rows > 9223372036854775807L / group_count ||
    rows * group_count > 9223372036854775807L / output_rank ||
    rows * group_count * output_rank > 9223372036854775807L / 2L {
    return false
  }
  let input_values = rows * head_count * width
  let weight_values = output_rank * head_count * width
  let output_values = rows * group_count * output_rank
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == input_values * 2L &&
  operands[1].alignment() == 16L &&
  operands[2].byte_count() == weight_values * 2L &&
  operands[2].alignment() == 16L &&
  operands[3].byte_count() == output_values * 2L &&
  operands[3].alignment() == 16L
}
