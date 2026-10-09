///|
pub fn lower_moe_grouped_routing_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeGroupedRoutingCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      MoeGroupedNoAuxiliaryLossTopK(
        routed_expert_count,
        group_count,
        group_score_topk,
        selected_group_count,
        experts_per_token,
        normalize,
        scaling
      ),
      MoeGroupedRoutingAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if group_count != 1 ||
    group_score_topk != 2 ||
    selected_group_count != 1 ||
    experts_per_token != 8 ||
    !normalize ||
    scaling != 2.5 {
    raise Invalid(Shape)
  }
  let family = match
    (requirements.geometry().hidden_size(), routed_expert_count) {
    (6144, 256) => FullGroupedRouting256
    (4096, 288) => FlashGroupedRouting288
    _ => raise Invalid(Shape)
  }
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() {
    raise Invalid(Limit)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  let score_cells = checked_mul(
    maximum_tokens.to_int64(),
    routed_expert_count.to_int64(),
  )
  let output_cells = checked_mul(maximum_tokens.to_int64(), 8L)
  let score_bytes = checked_mul(score_cells, 4L)
  let output_bytes = checked_mul(output_cells, 4L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(MoeUncorrectedScoresOutput, score_bytes, 16L),
    dense_projection_operand(MoeChoiceScoresOutput, score_bytes, 16L),
    dense_projection_operand(MoeSelectedIndicesOutput, output_bytes, 4L),
    dense_projection_operand(MoeSelectedWeightsOutput, output_bytes, 16L),
  ]
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let rendered = @grouped_routing_cuda_source.grouped_choice_and_selected_weight_f32_source(
    routed_expert_count~,
    group_count=1,
    group_score_topk=2,
    selected_group_count=1,
    experts_per_token=8,
    normalize=true,
    scaling=2.5,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_moe_grouped_routing_f32_v1",
  ) catch {
    error => raise GroupedRoutingSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let family_tag = match family {
    FullGroupedRouting256 => "full-experts256"
    FlashGroupedRouting288 => "flash-experts288"
  }
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-moe-grouped-routing-cuda-aot.v1" +
    "|family=" +
    family_tag +
    "|choice=row-major-f32-corrected-scores" +
    "|weights=row-major-f32-uncorrected-sigmoid-scores" +
    "|groups=1:group-score-top2:selected-groups1" +
    "|experts=top8|normalize=sum-plus-1e-20|scale=2.5" +
    "|tie-policy=lower-id-deterministic-pytorch-unsorted-topk-parity-unqualified" +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=expert-ffn-shared-expert-combine-residual" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|geometry=" +
    routed_expert_count.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    routed_expert_count,
    maximum_tokens,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: rendered.function_symbol(),
    source,
    source_digest,
    recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
