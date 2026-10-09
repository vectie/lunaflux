///|
fn exact_dense_swiglu_abi(
  geometry : @glm_hybrid_kernel_requirements.GlmHybridKernelGeometry,
  hidden_size : Int,
  intermediate_size : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if hidden_size != geometry.hidden_size() || intermediate_size <= 0 {
    return false
  }
  guard claim.operands is [counts, input, gate, up, down, output] else {
    return false
  }
  guard checked_positive_mul(hidden_size.to_int64(), 2L) is Some(row_bytes) &&
    exact_rows(input.byte_count, row_bytes) is Some(rows) &&
    checked_positive_mul(hidden_size.to_int64(), intermediate_size.to_int64())
    is Some(weight_elements) &&
    checked_positive_mul(weight_elements, 2L) is Some(weight_bytes) &&
    checked_positive_mul(rows, hidden_size.to_int64()) is Some(work) else {
    return false
  }
  claim.function_symbol == "lunaflux_glm_hybrid_dense_swiglu_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, HiddenInput, input.byte_count, 16L) &&
  operand_is(gate, DenseGateWeights, weight_bytes, 16L) &&
  operand_is(up, DenseUpWeights, weight_bytes, 16L) &&
  operand_is(down, DenseDownWeights, weight_bytes, 16L) &&
  operand_is(output, HiddenOutput, input.byte_count, 16L) &&
  exact_launch(claim.launch, work, 0)
}
