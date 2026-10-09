///|
fn exact_moe_decoder_residual_add_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  hidden_width : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if hidden_width != requirements.geometry().hidden_size() {
    return false
  }
  guard claim.operands is [counts, residual, branch, output] else {
    return false
  }
  let hidden = hidden_width.to_int64()
  guard checked_positive_mul(hidden, 2L) is Some(row_bytes) &&
    exact_rows(residual.byte_count, row_bytes) is Some(rows) &&
    checked_positive_mul(rows, hidden) is Some(work) else {
    return false
  }
  rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol ==
  "lunaflux_glm_hybrid_moe_decoder_residual_add_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(residual, DecoderResidualInput, residual.byte_count, 16L) &&
  operand_is(branch, MoeCombinedOutput, residual.byte_count, 16L) &&
  operand_is(output, DecoderResidualOutput, residual.byte_count, 16L) &&
  exact_launch(claim.launch, work, 0)
}

///|
fn exact_hyper_connection_mix_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  stream_count : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if stream_count <= 0 {
    return false
  }
  guard claim.operands is [counts, post, combine, residual, branch, output] else {
    return false
  }
  let hidden = requirements.geometry().hidden_size().to_int64()
  let streams = stream_count.to_int64()
  guard checked_positive_mul(hidden, 2L) is Some(branch_row_bytes) &&
    checked_positive_mul(streams, 4L) is Some(post_row_bytes) &&
    checked_positive_mul(streams, streams) is Some(matrix_cells) &&
    checked_positive_mul(matrix_cells, 4L) is Some(combine_row_bytes) &&
    checked_positive_mul(streams, branch_row_bytes) is Some(stream_row_bytes) &&
    exact_rows(branch.byte_count, branch_row_bytes) is Some(rows) &&
    checked_positive_mul(rows, post_row_bytes) is Some(post_bytes) &&
    checked_positive_mul(rows, combine_row_bytes) is Some(combine_bytes) &&
    checked_positive_mul(rows, stream_row_bytes) is Some(stream_bytes) &&
    checked_positive_mul(rows, streams) is Some(row_streams) &&
    checked_positive_mul(row_streams, hidden) is Some(work) else {
    return false
  }
  rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_hyper_connection_mix_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(post, HyperPostControlInput, post_bytes, 16L) &&
  operand_is(combine, HyperCombineControlInput, combine_bytes, 16L) &&
  operand_is(residual, HyperResidualStreamsInput, stream_bytes, 16L) &&
  operand_is(branch, HyperSublayerOutputInput, branch.byte_count, 16L) &&
  operand_is(output, HyperMixedStreamsOutput, stream_bytes, 16L) &&
  exact_launch(claim.launch, work, 0)
}
