///|
fn exact_dsa_a_projection_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  input_width : Int,
  output_width : Int,
  input_is_hidden : Bool,
  input_role : GlmHybridLaunchOperandRole,
  output_role : GlmHybridLaunchOperandRole,
  function_symbol : String,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if (input_is_hidden && input_width != geometry.hidden_size()) ||
    output_width <= 0 {
    return false
  }
  guard claim.operands is [counts, input, weights, output] else { return false }
  guard checked_positive_mul(input_width.to_int64(), 2L)
    is Some(input_row_bytes) else {
    return false
  }
  guard checked_positive_mul(output_width.to_int64(), 2L)
    is Some(output_row_bytes) else {
    return false
  }
  guard exact_rows(input.byte_count, input_row_bytes) is Some(rows) else {
    return false
  }
  guard checked_positive_mul(rows, output_row_bytes) is Some(output_bytes) else {
    return false
  }
  guard checked_positive_mul(input_width.to_int64(), output_width.to_int64())
    is Some(weight_elements) else {
    return false
  }
  guard checked_positive_mul(weight_elements, 2L) is Some(weight_bytes) else {
    return false
  }
  guard checked_positive_mul(rows, output_width.to_int64()) is Some(work) else {
    return false
  }
  claim.function_symbol == function_symbol &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, input_role, input.byte_count, 16L) &&
  operand_is(weights, ModelWeights, weight_bytes, 16L) &&
  operand_is(output, output_role, output_bytes, 16L) &&
  exact_launch(claim.launch, work, 0)
}

///|
fn exact_dsa_query_a_norm_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  width : Int,
  epsilon : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  ignore(geometry)
  if width <= 0 || epsilon <= 0.0 {
    return false
  }
  guard claim.operands is [counts, input, weights, output] else { return false }
  guard checked_positive_mul(width.to_int64(), 2L) is Some(row_bytes) else {
    return false
  }
  guard exact_rows(input.byte_count, row_bytes) is Some(rows) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_query_a_rms_norm_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, DsaQueryCompressedInput, input.byte_count, 16L) &&
  operand_is(weights, ModelWeights, row_bytes, 16L) &&
  operand_is(output, DsaQueryNormalizedOutput, input.byte_count, 16L) &&
  claim.launch.grid_x() == rows.to_int() &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 1024
}

///|
fn exact_dsa_kv_split_norm_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  epsilon : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  let mut geometry : (Int, Int)? = None
  for item in requirements.requirements() {
    if item.operation() is DsaKeyValueAProjection(_, prefix, suffix) {
      if geometry is Some(_) {
        return false
      }
      geometry = Some((prefix, suffix))
    }
  }
  guard geometry is Some((prefix_width, suffix_width)) else { return false }
  if prefix_width <= 0 || suffix_width < 0 || epsilon <= 0.0 {
    return false
  }
  let core = match claim.operands {
    [counts, input, weights, normalized] if suffix_width == 0 =>
      Some((counts, input, weights, normalized, None))
    [counts, input, weights, normalized, suffix] if suffix_width > 0 =>
      Some((counts, input, weights, normalized, Some(suffix)))
    _ => None
  }
  guard core is Some((counts, input, weights, normalized, suffix)) else {
    return false
  }
  guard checked_positive_mul((prefix_width + suffix_width).to_int64(), 2L)
    is Some(input_row_bytes) else {
    return false
  }
  guard exact_rows(input.byte_count, input_row_bytes) is Some(rows) else {
    return false
  }
  let prefix_bytes = rows * prefix_width.to_int64() * 2L
  let suffix_ok = match suffix {
    None => suffix_width == 0
    Some(output) =>
      operand_is(
        output,
        DsaKeyRopeSuffixOutput,
        rows * suffix_width.to_int64() * 2L,
        16L,
      )
  }
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_kv_a_split_rms_norm_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, DsaKeyValueCompressedRopeInput, input.byte_count, 16L) &&
  operand_is(weights, ModelWeights, prefix_width.to_int64() * 2L, 16L) &&
  operand_is(normalized, DsaKeyValueNormalizedOutput, prefix_bytes, 16L) &&
  suffix_ok &&
  claim.launch.grid_x() == rows.to_int() &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 1024
}
