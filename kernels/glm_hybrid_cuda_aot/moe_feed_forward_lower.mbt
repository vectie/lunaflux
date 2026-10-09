///|
pub fn lower_moe_routed_expert_feed_forward_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      RoutedExpertFeedForward(expert_count, intermediate_size, activation),
      Advanced(RoutedExpertFeedForwardAot),
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard moe_feed_forward_selected_width(requirements, expert_count)
    is Some(experts_per_token) else {
    raise Invalid(Requirement)
  }
  let hidden_size = requirements.geometry().hidden_size()
  let (family, source_activation, activation_tag) = match
    (
      hidden_size, expert_count, intermediate_size, activation, experts_per_token,
    ) {
    (6144, 256, 2048, SwiGlu, 8) =>
      (
        FullRoutedExperts256,
        @selected_expert_mlp_cuda_source.StandardSwiGlu,
        "swiglu",
      )
    (4096, 288, 2048, ClampedSwiGlu(10.0), 8) =>
      (FlashRoutedExperts288, BoundedClampedSwiGlu(10.0), "clamped-swiglu-10")
    _ => raise Invalid(Shape)
  }
  validate_moe_feed_forward_environment(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let row_cells = checked_mul(maximum_tokens.to_int64(), hidden_size.to_int64())
  let selected_cells = checked_mul(
    maximum_tokens.to_int64(),
    experts_per_token.to_int64(),
  )
  let bank_cells = checked_mul(
    checked_mul(expert_count.to_int64(), intermediate_size.to_int64()),
    hidden_size.to_int64(),
  )
  let row_bytes = checked_mul(row_cells, 2L)
  let selected_bytes = checked_mul(selected_cells, 4L)
  let bank_bytes = checked_mul(bank_cells, 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(MoeFeedForwardInput, row_bytes, 16L),
    dense_projection_operand(MoeSelectedIndicesOutput, selected_bytes, 4L),
    dense_projection_operand(MoeSelectedWeightsOutput, selected_bytes, 16L),
    dense_projection_operand(MoeExpertGateWeights, bank_bytes, 16L),
    dense_projection_operand(MoeExpertUpWeights, bank_bytes, 16L),
    dense_projection_operand(MoeExpertDownWeights, bank_bytes, 16L),
    dense_projection_operand(MoeRoutedOutput, row_bytes, 16L),
  ]
  let rendered = @selected_expert_mlp_cuda_source.selected_expert_swiglu_bf16_source(
    hidden_size~,
    intermediate_size~,
    routed_expert_count=expert_count,
    experts_per_token~,
    activation=source_activation,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_moe_routed_experts_bf16_v1",
  ) catch {
    error => raise SelectedExpertMlpSource(error)
  }
  moe_feed_forward_candidate(
    family,
    SelectedExpertBf16WeightedAccumulate,
    requirements,
    requirement,
    requirement_ordinal,
    maximum_tokens,
    target,
    moe_feed_forward_dimensions(maximum_tokens, hidden_size),
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    "routed-selected-expert-" + activation_tag,
    hidden_size.to_string() +
    ":" +
    intermediate_size.to_string() +
    ":" +
    expert_count.to_string() +
    ":" +
    experts_per_token.to_string() +
    ":" +
    maximum_tokens.to_string(),
    "ordered-f32-accumulation-expf-bf16-stage-rounds",
    "routing-shared-expert-routed-shared-add-decoder-residual-expert-parallelism",
    limits,
  )
}

///|
pub fn lower_moe_shared_expert_feed_forward_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      SharedExpertFeedForward(shared_count, intermediate_size, activation),
      Advanced(SharedExpertFeedForwardAot),
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let hidden_size = requirements.geometry().hidden_size()
  let (family, activation_tag) = match
    (hidden_size, shared_count, intermediate_size, activation) {
    (6144, 1, 2048, SwiGlu) => (FullSharedExpert, "swiglu")
    (4096, 1, 2048, ClampedSwiGlu(10.0)) =>
      (FlashSharedExpert, "clamped-swiglu-10")
    _ => raise Invalid(Shape)
  }
  validate_moe_feed_forward_environment(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let row_cells = checked_mul(maximum_tokens.to_int64(), hidden_size.to_int64())
  let weight_cells = checked_mul(
    hidden_size.to_int64(),
    intermediate_size.to_int64(),
  )
  let row_bytes = checked_mul(row_cells, 2L)
  let weight_bytes = checked_mul(weight_cells, 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(MoeFeedForwardInput, row_bytes, 16L),
    dense_projection_operand(MoeSharedGateWeights, weight_bytes, 16L),
    dense_projection_operand(MoeSharedUpWeights, weight_bytes, 16L),
    dense_projection_operand(MoeSharedDownWeights, weight_bytes, 16L),
    dense_projection_operand(MoeSharedOutput, row_bytes, 16L),
  ]
  let rendered = match activation {
    SwiGlu =>
      @gated_mlp_cuda_source.separate_gate_up_swiglu_dense_bf16_source(
        hidden_size~,
        intermediate_size~,
        maximum_rows=maximum_tokens,
        function_symbol="lunaflux_glm_hybrid_moe_shared_expert_bf16_v1",
      ) catch {
        error => raise GatedMlpSource(error)
      }
    ClampedSwiGlu(limit) =>
      @gated_mlp_cuda_source.separate_gate_up_clamped_swiglu_dense_bf16_source(
        hidden_size~,
        intermediate_size~,
        maximum_rows=maximum_tokens,
        clamp_limit=limit,
        function_symbol="lunaflux_glm_hybrid_moe_shared_expert_bf16_v1",
      ) catch {
        error => raise GatedMlpSource(error)
      }
  }
  moe_feed_forward_candidate(
    family,
    SharedExpertBf16SwiGlu,
    requirements,
    requirement,
    requirement_ordinal,
    maximum_tokens,
    target,
    moe_feed_forward_dimensions(maximum_tokens, hidden_size),
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    "shared-expert-" + activation_tag,
    hidden_size.to_string() +
    ":" +
    intermediate_size.to_string() +
    ":1:" +
    maximum_tokens.to_string(),
    "ordered-f32-accumulation-expf-bf16-stage-rounds",
    "routing-routed-experts-routed-shared-add-decoder-residual",
    limits,
  )
}

///|
pub fn lower_moe_routed_shared_combine_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      RoutedSharedExpertCombine(experts_per_token, normalize, scaling),
      Advanced(RoutedSharedExpertCombineAot),
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let hidden_size = requirements.geometry().hidden_size()
  let family = match (hidden_size, experts_per_token, normalize, scaling) {
    (6144, 8, true, 2.5) => FullRoutedSharedAdd
    (4096, 8, true, 2.5) => FlashRoutedSharedAdd
    _ => raise Invalid(Shape)
  }
  validate_moe_feed_forward_environment(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let row_cells = checked_mul(maximum_tokens.to_int64(), hidden_size.to_int64())
  let row_bytes = checked_mul(row_cells, 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(MoeRoutedOutput, row_bytes, 16L),
    dense_projection_operand(MoeSharedOutput, row_bytes, 16L),
    dense_projection_operand(MoeCombinedOutput, row_bytes, 16L),
  ]
  let rendered = @bf16_binary_cuda_source.bf16_add_source(
    width=hidden_size,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_moe_routed_shared_add_bf16_v1",
  ) catch {
    error => raise Bf16BinarySource(error)
  }
  moe_feed_forward_candidate(
    family,
    RoutedSharedF32AddBf16Round,
    requirements,
    requirement,
    requirement_ordinal,
    maximum_tokens,
    target,
    moe_feed_forward_dimensions(maximum_tokens, hidden_size),
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    "routed-plus-shared-bf16-add",
    hidden_size.to_string() + ":8:true:2.5:" + maximum_tokens.to_string(),
    "f32-add-one-bf16-round",
    "routing-expert-ffn-shared-ffn-decoder-residual",
    limits,
  )
}
