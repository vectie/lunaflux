///|
fn exact_glm_interleaved_rotary_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  theta : Double?,
  interleaved : Bool,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  guard theta is Some(theta_value) else { return false }
  if theta_value.is_nan() ||
    theta_value.is_inf() ||
    theta_value <= 0.0 ||
    !interleaved {
    return false
  }
  let mut query_geometry : (Int, Int)? = None
  let mut rope_width : Int? = None
  for item in requirements.requirements() {
    match item.operation() {
      DsaQueryBProjection(_, heads, dimension) => {
        if query_geometry is Some(_) {
          return false
        }
        query_geometry = Some((heads, dimension))
      }
      DsaKeyValueAProjection(_, _, suffix) => {
        if rope_width is Some(_) {
          return false
        }
        rope_width = Some(suffix)
      }
      _ => ()
    }
  }
  guard query_geometry is Some((query_heads, query_dimension)) &&
    rope_width is Some(rope_dimension) else {
    return false
  }
  if query_heads <= 0 ||
    query_dimension <= 0 ||
    rope_dimension <= 0 ||
    rope_dimension % 2 != 0 ||
    rope_dimension > query_dimension {
    return false
  }
  guard claim.operands
    is [counts, positions, query_input, key_input, query_output, key_output] else {
    return false
  }
  guard checked_positive_mul(query_heads.to_int64(), query_dimension.to_int64())
    is Some(query_row_elements) &&
    checked_positive_mul(query_row_elements, 2L) is Some(query_row_bytes) &&
    exact_rows(query_input.byte_count, query_row_bytes) is Some(rows) &&
    checked_positive_mul(rows, 4L) is Some(position_bytes) &&
    checked_positive_mul(rope_dimension.to_int64(), 2L) is Some(key_row_bytes) &&
    checked_positive_mul(rows, key_row_bytes) is Some(key_bytes) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_interleaved_rope_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(positions, QueryPositionsInput, position_bytes, 4L) &&
  operand_is(query_input, DsaQueryHeadsOutput, query_input.byte_count, 16L) &&
  operand_is(key_input, DsaKeyRopeSuffixOutput, key_bytes, 16L) &&
  operand_is(
    query_output,
    DsaAttentionQueriesInput,
    query_input.byte_count,
    16L,
  ) &&
  operand_is(key_output, DsaAttentionKeyRopeInput, key_bytes, 16L) &&
  claim.launch.grid_x() == rows.to_int() &&
  claim.launch.grid_y() == query_heads + 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 256 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
