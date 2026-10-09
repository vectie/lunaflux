///|
fn exact_kda_decay_control_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  head_count : Int,
  head_dimension : Int,
  lower_bound : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if head_count != 64 || head_dimension != 128 || lower_bound != -5.0 {
    return false
  }
  guard claim.operands is [counts, raw, bias, log_rate, output] else {
    return false
  }
  guard checked_positive_mul(head_count.to_int64(), head_dimension.to_int64())
    is Some(row_width) &&
    exact_rows(raw.byte_count, row_width * 2L) is Some(maximum_rows) &&
    checked_positive_mul(maximum_rows, row_width * 4L) is Some(output_bytes) else {
    return false
  }
  maximum_rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_kda_decay_control_f32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(raw, KdaDecayRawForgetInput, raw.byte_count, 16L) &&
  operand_is(bias, KdaDecayBiasInput, row_width * 4L, 16L) &&
  operand_is(log_rate, KdaDecayLogRateInput, head_count.to_int64() * 4L, 16L) &&
  operand_is(output, KdaDecayOutput, output_bytes, 16L) &&
  claim.launch.grid_x().to_int64() == (maximum_rows * row_width + 255L) / 256L &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
