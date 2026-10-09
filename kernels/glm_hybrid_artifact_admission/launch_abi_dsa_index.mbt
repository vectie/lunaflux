///|
fn dsa_pooling_policy(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Bool)? {
  let mut result : (Int, Bool)? = None
  for requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexPooling(pool_size, true, tail) {
      if result is Some(_) {
        return None
      }
      result = Some((pool_size, tail))
    }
  }
  match result {
    Some(value) => Some(value)
    None => Some((1, false))
  }
}

///|
fn dsa_index_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int, Int, Bool)? {
  let mut result = None
  for requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexCompute(a, b, c, d) {
      if result is Some(_) {
        return None
      }
      result = Some((a, b, c, d))
    }
  }
  result
}

///|
fn exact_dsa_index_compute_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  head_count : Int,
  head_dimension : Int,
  top_k : Int,
  rope_interleave : Bool,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if head_count <= 0 || head_dimension <= 0 || top_k <= 0 || !rope_interleave {
    return false
  }
  guard dsa_pooling_policy(requirements) is Some((pool_size, append_tail)) else {
    return false
  }
  if pool_size <= 0 || top_k % pool_size != 0 {
    return false
  }
  guard claim.operands
    is [
      counts,
      positions,
      queries,
      head_weights,
      candidate_keys,
      raw_indices,
      candidate_valid,
      candidate_count,
      key_valid,
      output,
    ] else {
    return false
  }
  guard exact_rows(positions.byte_count, 4L) is Some(maximum_tokens) else {
    return false
  }
  if maximum_tokens < pool_size.to_int64() ||
    maximum_tokens > requirements.geometry().context_length().to_int64() {
    return false
  }
  let candidate_capacity = maximum_tokens / pool_size.to_int64()
  let output_width = top_k.to_int64() +
    (if append_tail { pool_size.to_int64() - 1L } else { 0L })
  guard checked_positive_mul(maximum_tokens, head_count.to_int64())
    is Some(query_heads) else {
    return false
  }
  guard checked_positive_mul(query_heads, head_dimension.to_int64())
    is Some(query_elements) else {
    return false
  }
  guard checked_positive_mul(candidate_capacity, head_dimension.to_int64())
    is Some(candidate_elements) else {
    return false
  }
  guard checked_positive_mul(candidate_capacity, pool_size.to_int64())
    is Some(raw_index_elements) else {
    return false
  }
  guard checked_positive_mul(maximum_tokens, output_width)
    is Some(output_elements) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_index_select_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(positions, QueryPositionsInput, maximum_tokens * 4L, 4L) &&
  operand_is(queries, DsaIndexQueriesInput, query_elements * 2L, 16L) &&
  operand_is(head_weights, DsaIndexHeadWeightsInput, query_heads * 2L, 16L) &&
  operand_is(
    candidate_keys,
    DsaCandidateKeysInput,
    candidate_elements * 2L,
    16L,
  ) &&
  operand_is(
    raw_indices,
    DsaCandidateRawIndicesInput,
    raw_index_elements * 4L,
    4L,
  ) &&
  operand_is(
    candidate_valid,
    DsaCandidateValidityInput,
    candidate_capacity * 4L,
    4L,
  ) &&
  operand_is(candidate_count, DsaCandidateCountInput, 4L, 4L) &&
  operand_is(key_valid, DsaKeyValidityInput, maximum_tokens * 4L, 4L) &&
  operand_is(output, DsaSelectedIndicesOutput, output_elements * 4L, 4L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}

///|
fn exact_dsa_index_reuse_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  guard dsa_pooling_policy(requirements) is Some((pool_size, append_tail)) else {
    return false
  }
  guard dsa_index_geometry(requirements) is Some((_, _, top_k, true)) else {
    return false
  }
  let output_width = top_k.to_int64() +
    (if append_tail { pool_size.to_int64() - 1L } else { 0L })
  guard claim.operands is [counts, input, output] else { return false }
  guard checked_positive_mul(output_width, 4L) is Some(row_bytes) else {
    return false
  }
  guard exact_rows(input.byte_count, row_bytes) is Some(maximum_tokens) else {
    return false
  }
  guard checked_positive_mul(maximum_tokens, output_width) is Some(work) else {
    return false
  }
  maximum_tokens <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_index_reuse_i32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, DsaSelectedIndicesInput, input.byte_count, 4L) &&
  operand_is(output, DsaSelectedIndicesOutput, input.byte_count, 4L) &&
  exact_launch(claim.launch, work, 0)
}
