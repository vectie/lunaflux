///|
fn moe_selected_width(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  routed_expert_count : Int,
) -> Int? {
  for requirement in requirements.requirements() {
    if requirement.operation()
      is MoeGroupedNoAuxiliaryLossTopK(
        expert_count,
        _,
        _,
        _,
        experts_per_token,
        _,
        _
      ) &&
      expert_count == routed_expert_count {
      return Some(experts_per_token)
    }
  }
  None
}

///|
fn exact_moe_routed_expert_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  routed_expert_count : Int,
  intermediate_size : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  guard moe_selected_width(requirements, routed_expert_count)
    is Some(experts_per_token) else {
    return false
  }
  guard claim.operands
    is [counts, input, indices, weights, gate, up, down, output] else {
    return false
  }
  let hidden = requirements.geometry().hidden_size().to_int64()
  let intermediate = intermediate_size.to_int64()
  let experts = routed_expert_count.to_int64()
  guard checked_positive_mul(hidden, 2L) is Some(row_bytes) &&
    checked_positive_mul(experts_per_token.to_int64(), 4L)
    is Some(selected_row_bytes) &&
    exact_rows(input.byte_count, row_bytes) is Some(rows) &&
    checked_positive_mul(rows, selected_row_bytes) is Some(selected_bytes) &&
    checked_positive_mul(experts, intermediate) is Some(expert_intermediate) &&
    checked_positive_mul(expert_intermediate, hidden) is Some(weight_cells) &&
    checked_positive_mul(weight_cells, 2L) is Some(weight_bytes) &&
    checked_positive_mul(rows, hidden) is Some(work) else {
    return false
  }
  rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_moe_routed_experts_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, MoeFeedForwardInput, input.byte_count, 16L) &&
  operand_is(indices, MoeSelectedIndicesOutput, selected_bytes, 4L) &&
  operand_is(weights, MoeSelectedWeightsOutput, selected_bytes, 16L) &&
  operand_is(gate, MoeExpertGateWeights, weight_bytes, 16L) &&
  operand_is(up, MoeExpertUpWeights, weight_bytes, 16L) &&
  operand_is(down, MoeExpertDownWeights, weight_bytes, 16L) &&
  operand_is(output, MoeRoutedOutput, input.byte_count, 16L) &&
  exact_launch(claim.launch, work, 0)
}

///|
fn exact_moe_shared_expert_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  shared_expert_count : Int,
  intermediate_size : Int,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if shared_expert_count != 1 {
    return false
  }
  guard claim.operands is [counts, input, gate, up, down, output] else {
    return false
  }
  let hidden = requirements.geometry().hidden_size().to_int64()
  let intermediate = intermediate_size.to_int64()
  guard checked_positive_mul(hidden, 2L) is Some(row_bytes) &&
    exact_rows(input.byte_count, row_bytes) is Some(rows) &&
    checked_positive_mul(hidden, intermediate) is Some(weight_cells) &&
    checked_positive_mul(weight_cells, 2L) is Some(weight_bytes) &&
    checked_positive_mul(rows, hidden) is Some(work) else {
    return false
  }
  rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_moe_shared_expert_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(input, MoeFeedForwardInput, input.byte_count, 16L) &&
  operand_is(gate, MoeSharedGateWeights, weight_bytes, 16L) &&
  operand_is(up, MoeSharedUpWeights, weight_bytes, 16L) &&
  operand_is(down, MoeSharedDownWeights, weight_bytes, 16L) &&
  operand_is(output, MoeSharedOutput, input.byte_count, 16L) &&
  exact_launch(claim.launch, work, 0)
}

///|
fn exact_moe_routed_shared_combine_abi(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  experts_per_token : Int,
  normalize : Bool,
  scaling : Double,
  claim : GlmHybridArtifactClaimInput,
) -> Bool {
  if experts_per_token <= 0 || !normalize || scaling <= 0.0 {
    return false
  }
  guard claim.operands is [counts, routed, shared, output] else { return false }
  let hidden = requirements.geometry().hidden_size().to_int64()
  guard checked_positive_mul(hidden, 2L) is Some(row_bytes) &&
    exact_rows(routed.byte_count, row_bytes) is Some(rows) &&
    checked_positive_mul(rows, hidden) is Some(work) else {
    return false
  }
  rows <= requirements.geometry().context_length().to_int64() &&
  claim.function_symbol == "lunaflux_glm_hybrid_moe_routed_shared_add_bf16_v1" &&
  operand_is(counts, StepCountsInput, 20L, 4L) &&
  operand_is(routed, MoeRoutedOutput, routed.byte_count, 16L) &&
  operand_is(shared, MoeSharedOutput, routed.byte_count, 16L) &&
  operand_is(output, MoeCombinedOutput, routed.byte_count, 16L) &&
  exact_launch(claim.launch, work, 0)
}
