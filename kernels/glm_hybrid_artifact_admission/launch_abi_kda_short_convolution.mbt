///|
fn exact_kda_short_convolution_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  channel_count : Int,
  kernel_size : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if channel_count != 8192 || kernel_size != 4 {
    return false
  }
  guard claim.operands
    is [counts, offsets, input, weights, initial, output, final_] else {
    return false
  }
  guard checked_positive_mul(channel_count.to_int64(), 2L) is Some(row_bytes) &&
    exact_rows(input.byte_count, row_bytes) is Some(maximum_rows) &&
    exact_rows(offsets.byte_count, 4L) is Some(offset_cells) else {
    return false
  }
  let maximum_sequences = offset_cells - 1L
  if maximum_sequences <= 0L ||
    maximum_sequences > maximum_rows ||
    maximum_rows > requirements.geometry().context_length().to_int64() {
    return false
  }
  let history = kernel_size.to_int64() - 1L
  guard checked_positive_mul(channel_count.to_int64(), kernel_size.to_int64())
    is Some(weight_elements) &&
    checked_positive_mul(maximum_sequences, channel_count.to_int64())
    is Some(sequence_channels) &&
    checked_positive_mul(sequence_channels, history) is Some(state_elements) &&
    checked_positive_mul(maximum_rows, channel_count.to_int64())
    is Some(output_work) &&
    checked_positive_mul(weight_elements, 2L) is Some(weight_bytes) &&
    checked_positive_mul(state_elements, 2L) is Some(state_bytes) else {
    return false
  }
  let work = if output_work > state_elements {
    output_work
  } else {
    state_elements
  }
  claim.function_symbol == "lunaflux_glm_hybrid_kda_short_convolution_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(offsets, KdaSequenceOffsetsInput, offset_cells * 4L, 4L) &&
  operand_is(input, KdaProjectedInput, input.byte_count, 16L) &&
  operand_is(weights, KdaShortConvolutionWeights, weight_bytes, 16L) &&
  operand_is(initial, KdaConvolutionInitialStateInput, state_bytes, 16L) &&
  operand_is(output, KdaConvolvedOutput, input.byte_count, 16L) &&
  operand_is(final_, KdaConvolutionFinalStateOutput, state_bytes, 16L) &&
  exact_launch(claim.launch, work, 0)
}
