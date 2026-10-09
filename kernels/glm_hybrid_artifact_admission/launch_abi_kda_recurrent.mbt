///|
fn exact_kda_recurrent_update_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  head_count : Int,
  head_dimension : Int,
  gate_lower_bound : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if head_count != 64 || head_dimension != 128 || gate_lower_bound != -5.0 {
    return false
  }
  guard claim.operands
    is [
      counts,
      offsets,
      query,
      key,
      value,
      log_decay,
      beta,
      initial_state,
      output,
      final_state,
    ] else {
    return false
  }
  guard checked_positive_mul(head_count.to_int64(), head_dimension.to_int64())
    is Some(row_width) &&
    checked_positive_mul(row_width, 2L) is Some(bf16_row_bytes) &&
    exact_rows(query.byte_count, bf16_row_bytes) is Some(maximum_rows) &&
    exact_rows(offsets.byte_count, 4L) is Some(offset_cells) else {
    return false
  }
  let maximum_sequences = offset_cells - 1L
  if maximum_sequences <= 0L ||
    maximum_sequences > maximum_rows ||
    maximum_rows > requirements.geometry().context_length().to_int64() {
    return false
  }
  guard checked_positive_mul(maximum_rows, row_width) is Some(vector_cells) &&
    checked_positive_mul(vector_cells, 2L) is Some(vector_bytes) &&
    checked_positive_mul(vector_cells, 4L) is Some(decay_bytes) &&
    checked_positive_mul(maximum_rows, head_count.to_int64())
    is Some(gate_cells) &&
    checked_positive_mul(gate_cells, 2L) is Some(gate_bytes) &&
    checked_positive_mul(row_width, head_dimension.to_int64())
    is Some(state_width) &&
    checked_positive_mul(maximum_sequences, state_width) is Some(state_cells) &&
    checked_positive_mul(state_cells, 4L) is Some(state_bytes) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_kda_recurrent_delta_bf16_f32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(offsets, KdaSequenceOffsetsInput, offset_cells * 4L, 4L) &&
  operand_is(query, KdaRecurrentQueryInput, vector_bytes, 16L) &&
  operand_is(key, KdaRecurrentKeyInput, vector_bytes, 16L) &&
  operand_is(value, KdaRecurrentValueInput, vector_bytes, 16L) &&
  operand_is(log_decay, KdaLogDecayInput, decay_bytes, 16L) &&
  operand_is(beta, KdaInputGateInput, gate_bytes, 16L) &&
  operand_is(initial_state, KdaRecurrentInitialStateInput, state_bytes, 16L) &&
  operand_is(output, KdaRecurrentOutput, vector_bytes, 16L) &&
  operand_is(final_state, KdaRecurrentFinalStateOutput, state_bytes, 16L) &&
  claim.launch.grid_x() == 1 &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
