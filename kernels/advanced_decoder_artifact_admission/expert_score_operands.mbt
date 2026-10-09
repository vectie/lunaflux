///|
fn expert_score_operands_are_exact(
  routed_expert_count : Int,
  operands : ReadOnlyArray[AdvancedDecoderLaunchOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
) -> Bool {
  if routed_expert_count <= 0 || dimensions.grid_x() <= 0 {
    return false
  }
  let row_bytes = routed_expert_count.to_int64() * 4L
  if row_bytes <= 0L ||
    dimensions.grid_x().to_int64() > 9223372036854775807L / row_bytes {
    return false
  }
  let expected_score_bytes = dimensions.grid_x().to_int64() * row_bytes
  operands[0].byte_count() == 20L &&
  operands[0].alignment() == 4L &&
  operands[1].byte_count() == expected_score_bytes &&
  operands[1].alignment() >= 4L &&
  operands[2].byte_count() == expected_score_bytes &&
  operands[2].alignment() >= 4L
}
