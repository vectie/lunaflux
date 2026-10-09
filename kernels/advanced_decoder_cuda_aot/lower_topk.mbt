///|
fn selection_recipe(
  family : String,
  numerical : String,
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  routed_expert_count : Int,
  top_k : Int,
  normalize : Bool,
  scaling : Double,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-selection-cuda-aot.v1" +
    "|family=" +
    family +
    "|numerical=" +
    numerical +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|tokens=" +
    maximum_tokens.to_string() +
    "|routed-experts=" +
    routed_expert_count.to_string() +
    "|top-k=" +
    top_k.to_string() +
    "|normalize=" +
    normalize.to_string() +
    "|scaling=" +
    scaling.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":" +
    dimensions.grid_y().to_string() +
    ":" +
    dimensions.grid_z().to_string() +
    ":" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.block_y().to_string() +
    ":" +
    dimensions.block_z().to_string() +
    ":" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    routing_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
fn selection_dimensions(
  maximum_tokens : Int,
) -> @launch_contract.AotLaunchDimensions raise AdvancedDecoderCudaAotError {
  @launch_contract.AotLaunchDimensions::new(
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
}

///|
fn validate_selection_policy(
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : AdvancedDecoderCudaAotLimits,
) -> Unit raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
}

///|
/// Lowers deterministic non-hash biased top-k selection. Ties choose the lower
/// expert id. This candidate does not gather or normalize selected weights.
pub fn lower_no_auxiliary_loss_top_k_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderRoutingCudaCandidate raise AdvancedDecoderCudaAotError {
  validate_selection_policy(maximum_tokens, target, compiler, limits)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) &&
    (requirement.operation(), requirement.capability())
    is (NoAuxiliaryLossTopK(routed_expert_count, top_k), StableTopKRoutingAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if routed_expert_count <= 0 ||
    top_k <= 0 ||
    top_k > routed_expert_count ||
    top_k > 256 {
    raise Invalid(Shape)
  }
  let scores_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), routed_expert_count.to_int64()),
    4L,
  )
  let bias_bytes = checked_mul(routed_expert_count.to_int64(), 4L)
  let indices_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), top_k.to_int64()),
    4L,
  )
  if scores_bytes > 2147483647L || indices_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = selection_dimensions(maximum_tokens)
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(RoutingScoresInput, scores_bytes, 4L),
    operand_with_alignment(RoutingSelectionBiasInput, bias_bytes, 4L),
    operand_with_alignment(RoutingIndicesOutput, indices_bytes, 4L),
  ]
  let function_symbol = "lunaflux_advanced_decoder_stable_biased_top_k_i32_v1"
  let rendered = @decoder_routing_cuda_source.stable_biased_top_k_i32_source(
    routed_expert_count~,
    experts_per_token=top_k,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    error => raise RoutingSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = selection_recipe(
    "no-auxiliary-loss-top-k-i32", "stable-lower-expert-id-top-k-v1", requirements,
    requirement_ordinal, maximum_tokens, routed_expert_count, top_k, false, 1.0,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: NoAuxiliaryLossTopKI32,
    numerical_contract: StableLowerExpertIdTopKV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    routed_expert_count,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}

///|
/// Lowers selected-score gathering, optional normalization, and scaling. Input
/// indices may be produced by stable top-k or token-hash lookup.
pub fn lower_selected_expert_weight_finalize_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderRoutingCudaCandidate raise AdvancedDecoderCudaAotError {
  validate_selection_policy(maximum_tokens, target, compiler, limits)
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) &&
    (requirement.operation(), requirement.capability())
    is (
      SelectedExpertWeightFinalize(
        routed_expert_count,
        top_k,
        normalize,
        scaling
      ),
      SelectedExpertWeightFinalizeAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if routed_expert_count <= 0 ||
    top_k <= 0 ||
    top_k > routed_expert_count ||
    top_k > 256 ||
    scaling.is_nan() ||
    scaling.is_inf() ||
    scaling <= 0.0 {
    raise Invalid(Shape)
  }
  let scores_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), routed_expert_count.to_int64()),
    4L,
  )
  let selected_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), top_k.to_int64()),
    4L,
  )
  if scores_bytes > 2147483647L || selected_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = selection_dimensions(maximum_tokens)
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(RoutingScoresInput, scores_bytes, 4L),
    operand_with_alignment(RoutingIndicesInput, selected_bytes, 4L),
    operand_with_alignment(RoutingWeightsOutput, selected_bytes, 4L),
  ]
  let function_symbol = "lunaflux_advanced_decoder_selected_expert_weights_f32_v1"
  let rendered = @decoder_routing_cuda_source.selected_expert_weight_finalize_f32_source(
    routed_expert_count~,
    experts_per_token=top_k,
    maximum_rows=maximum_tokens,
    normalize~,
    scaling~,
    function_symbol~,
  ) catch {
    error => raise RoutingSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = selection_recipe(
    "selected-expert-weight-finalize-f32", "ordered-f32-normalize-scale-v1", requirements,
    requirement_ordinal, maximum_tokens, routed_expert_count, top_k, normalize, scaling,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SelectedExpertWeightFinalizeF32,
    numerical_contract: OrderedF32NormalizeScaleV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    routed_expert_count,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
