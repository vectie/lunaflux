///|
fn exact_hyper_connection_function_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  stream_count : Int,
  hidden_width : Int,
  control_epsilon : Double,
  normalization_epsilon : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if stream_count != 4 ||
    hidden_width != geometry.hidden_size() ||
    hidden_width != 4096 ||
    control_epsilon != 0.000001 ||
    normalization_epsilon != 0.00001 {
    return false
  }
  guard claim.operands
    is [
      counts,
      streams,
      function_weights,
      base,
      scale,
      post,
      initial_combination,
      collapsed,
    ] else {
    return false
  }
  let streams_per_row = stream_count.to_int64() * hidden_width.to_int64()
  let mix_width = (2 + stream_count).to_int64() * stream_count.to_int64()
  guard exact_rows(streams.byte_count, streams_per_row * 2L) is Some(rows) &&
    checked_positive_mul(rows, stream_count.to_int64() * 4L) is Some(post_bytes) &&
    checked_positive_mul(
      rows,
      stream_count.to_int64() * stream_count.to_int64() * 4L,
    )
    is Some(matrix_bytes) &&
    checked_positive_mul(rows, hidden_width.to_int64() * 2L)
    is Some(collapsed_bytes) else {
    return false
  }
  rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_hc_function_bf16_f32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(streams, HyperResidualStreamsInput, streams.byte_count, 16L) &&
  operand_is(
    function_weights,
    HyperFunctionWeightsInput,
    mix_width * streams_per_row * 2L,
    16L,
  ) &&
  operand_is(base, HyperBaseInput, mix_width * 4L, 16L) &&
  operand_is(scale, HyperScaleInput, 12L, 16L) &&
  operand_is(post, HyperPostControlInput, post_bytes, 16L) &&
  operand_is(
    initial_combination,
    HyperInitialCombinationOutput,
    matrix_bytes,
    16L,
  ) &&
  operand_is(collapsed, HyperCollapsedOutput, collapsed_bytes, 16L) &&
  claim.launch.grid_x().to_int64() == rows &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}

///|
fn exact_hyper_connection_sinkhorn_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  stream_count : Int,
  iterations : Int,
  epsilon : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if stream_count != 4 || iterations != 20 || epsilon != 0.000001 {
    return false
  }
  guard claim.operands is [counts, initial_combination, combination] else {
    return false
  }
  let row_bytes = stream_count.to_int64() * stream_count.to_int64() * 4L
  guard exact_rows(initial_combination.byte_count, row_bytes) is Some(rows) &&
    checked_positive_mul(rows, row_bytes) is Some(matrix_bytes) else {
    return false
  }
  rows <= geometry.context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_hc_sinkhorn_f32_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(
    initial_combination,
    HyperInitialCombinationOutput,
    matrix_bytes,
    16L,
  ) &&
  operand_is(combination, HyperCombineControlInput, matrix_bytes, 16L) &&
  claim.launch.grid_x().to_int64() == rows &&
  claim.launch.grid_y() == 1 &&
  claim.launch.grid_z() == 1 &&
  claim.launch.block_x() == 1 &&
  claim.launch.block_y() == 1 &&
  claim.launch.block_z() == 1 &&
  claim.launch.shared_memory_bytes() == 0
}
