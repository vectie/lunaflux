///|
fn exact_kda_output_projection_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  input_width : Int,
  output_width : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if input_width != 8192 ||
    output_width != 4096 ||
    output_width != geometry.hidden_size() {
    return false
  }
  guard claim.operands is [counts, input, weights, output] else { return false }
  guard checked_positive_mul(input_width.to_int64(), 2L)
    is Some(input_row_bytes) &&
    exact_rows(input.byte_count, input_row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(maximum_rows, output_width.to_int64() * 2L)
    is Some(output_bytes) &&
    checked_positive_mul(input_width.to_int64(), output_width.to_int64())
    is Some(weight_cells) &&
    checked_positive_mul(weight_cells, 2L) is Some(weight_bytes) else {
    return false
  }
  maximum_rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_kda_output_projection_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, KdaOutputProjectionInput, input.byte_count, 16L) &&
  operand_is(weights, KdaOutputProjectionWeights, weight_bytes, 16L) &&
  operand_is(output, KdaOutputProjectionOutput, output_bytes, 16L) &&
  claim.launch.grid_x().to_int64() ==
  (maximum_rows * output_width.to_int64() + 255L) / 256L &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
