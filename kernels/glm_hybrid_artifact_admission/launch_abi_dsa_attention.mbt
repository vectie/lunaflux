///|
fn unique_dsa_kv_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int)? {
  let mut found = None
  for requirement in requirements.requirements() {
    if requirement.operation() is KeyValueLowRankProjection(_, heads, _, value) {
      if found is Some(_) {
        return None
      }
      found = Some((heads, value))
    }
  }
  found
}

///|
fn exact_dsa_sparse_attention_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  query_heads : Int,
  top_k : Int,
  query_key_dimension : Int,
  value_dimension : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  guard unique_dsa_kv_geometry(requirements) is Some((kv_heads, kv_value)) else {
    return false
  }
  guard dsa_pooling_policy(requirements) is Some((pool_size, append_tail)) else {
    return false
  }
  if query_heads <= 0 ||
    kv_heads <= 0 ||
    query_heads % kv_heads != 0 ||
    kv_value != value_dimension ||
    pool_size <= 0 {
    return false
  }
  guard claim.operands
    is [
      counts,
      positions,
      key_count,
      key_valid,
      queries,
      keys,
      values,
      selected,
      output,
    ] else {
    return false
  }
  guard exact_rows(positions.byte_count, 4L) is Some(maximum_tokens) else {
    return false
  }
  let selected_width = top_k.to_int64() +
    (if append_tail { pool_size.to_int64() - 1L } else { 0L })
  guard checked_positive_mul(maximum_tokens, query_heads.to_int64())
    is Some(query_rows) else {
    return false
  }
  guard checked_positive_mul(maximum_tokens, kv_heads.to_int64())
    is Some(key_rows) else {
    return false
  }
  guard checked_positive_mul(query_rows, query_key_dimension.to_int64())
    is Some(query_elements) else {
    return false
  }
  guard checked_positive_mul(key_rows, query_key_dimension.to_int64())
    is Some(key_elements) else {
    return false
  }
  guard checked_positive_mul(key_rows, value_dimension.to_int64())
    is Some(value_elements) else {
    return false
  }
  guard checked_positive_mul(maximum_tokens, selected_width)
    is Some(selected_elements) else {
    return false
  }
  guard checked_positive_mul(query_rows, value_dimension.to_int64())
    is Some(output_elements) else {
    return false
  }
  maximum_tokens <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_dsa_sparse_attention_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(positions, QueryPositionsInput, maximum_tokens * 4L, 4L) &&
  operand_is(key_count, DsaKeyCountInput, 4L, 4L) &&
  operand_is(key_valid, DsaKeyValidityInput, maximum_tokens * 4L, 4L) &&
  operand_is(queries, DsaAttentionQueriesInput, query_elements * 2L, 16L) &&
  operand_is(keys, DsaAttentionKeysInput, key_elements * 2L, 16L) &&
  operand_is(values, DsaAttentionValuesInput, value_elements * 2L, 16L) &&
  operand_is(selected, DsaSelectedIndicesInput, selected_elements * 4L, 4L) &&
  operand_is(output, DsaAttentionOutput, output_elements * 2L, 16L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
