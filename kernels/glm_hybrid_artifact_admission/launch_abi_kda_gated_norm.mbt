///|
fn exact_kda_gated_rms_norm_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  head_count : Int,
  head_dimension : Int,
  epsilon : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if head_count != 64 ||
    head_dimension != 128 ||
    head_count * head_dimension != 8192 ||
    epsilon != 0.00001 {
    return false
  }
  guard claim.operands is [counts, recurrent, gate, weight, output] else {
    return false
  }
  guard checked_positive_mul(head_count.to_int64(), head_dimension.to_int64())
    is Some(row_width) &&
    checked_positive_mul(row_width, 2L) is Some(row_bytes) &&
    exact_rows(recurrent.byte_count, row_bytes) is Some(maximum_rows) &&
    checked_positive_mul(maximum_rows, row_bytes) is Some(vector_bytes) &&
    checked_positive_mul(head_dimension.to_int64(), 2L) is Some(weight_bytes) else {
    return false
  }
  maximum_rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_kda_gated_rms_norm_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(recurrent, KdaGatedNormRecurrentInput, vector_bytes, 16L) &&
  operand_is(gate, KdaGatedNormGateInput, vector_bytes, 16L) &&
  operand_is(weight, KdaGatedNormWeights, weight_bytes, 16L) &&
  operand_is(output, KdaGatedNormOutput, vector_bytes, 16L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
