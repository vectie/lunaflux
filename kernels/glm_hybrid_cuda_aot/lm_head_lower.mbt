///|
fn dense_projection_operand(
  role : @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole,
  byte_count : Int64,
  alignment : Int64,
) -> @glm_hybrid_artifact_admission.GlmHybridLaunchOperand raise GlmHybridCudaAotError {
  @glm_hybrid_artifact_admission.GlmHybridLaunchOperand::new(
    role~,
    byte_count~,
    alignment~,
  ) catch {
    error => raise Operand(error)
  }
}

///|
fn dense_projection_operand_tag(
  role : @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole,
) -> String {
  match role {
    StepCountsInput => "step-counts"
    TokenIdsInput => "token-ids"
    HiddenInput => "hidden-input"
    ModelWeights => "model-weights"
    HiddenOutput => "hidden-output"
    LogitsOutput => "logits-output"
    DsaIndexKeysInput => "dsa-index-keys-input"
    DsaIndexGateScoresInput => "dsa-index-gate-scores-input"
    DsaIndexValidityInput => "dsa-index-validity-input"
    DsaPooledKeysOutput => "dsa-pooled-keys-output"
    DsaPooledRawIndicesOutput => "dsa-pooled-raw-indices-output"
    DsaPooledValidityOutput => "dsa-pooled-validity-output"
    DsaPooledCountOutput => "dsa-pooled-count-output"
    QueryPositionsInput => "query-positions-input"
    DsaIndexQueriesInput => "dsa-index-queries-input"
    DsaIndexHeadWeightsInput => "dsa-index-head-weights-input"
    DsaCandidateKeysInput => "dsa-candidate-keys-input"
    DsaCandidateRawIndicesInput => "dsa-candidate-raw-indices-input"
    DsaCandidateValidityInput => "dsa-candidate-validity-input"
    DsaCandidateCountInput => "dsa-candidate-count-input"
    DsaKeyValidityInput => "dsa-key-validity-input"
    DsaSelectedIndicesInput => "dsa-selected-indices-input"
    DsaSelectedIndicesOutput => "dsa-selected-indices-output"
    DsaKeyCountInput => "dsa-key-count-input"
    DsaAttentionQueriesInput => "dsa-attention-queries-input"
    DsaAttentionKeysInput => "dsa-attention-keys-input"
    DsaAttentionValuesInput => "dsa-attention-values-input"
    DsaAttentionOutput => "dsa-attention-output"
    DsaQueryCompressedOutput => "dsa-query-compressed-output"
    DsaKeyValueCompressedRopeOutput => "dsa-key-value-compressed-rope-output"
    DsaQueryCompressedInput => "dsa-query-compressed-input"
    DsaQueryNormalizedOutput => "dsa-query-normalized-output"
    DsaQueryHeadsOutput => "dsa-query-heads-output"
    DsaQueryNormalizedInput => "dsa-query-normalized-input"
    DsaKeyValueNormalizedInput => "dsa-key-value-normalized-input"
    DsaKeyValueHeadsOutput => "dsa-key-value-heads-output"
    DsaKeyValueCompressedRopeInput => "dsa-key-value-compressed-rope-input"
    DsaKeyValueNormalizedOutput => "dsa-key-value-normalized-output"
    DsaKeyRopeSuffixOutput => "dsa-key-rope-suffix-output"
    DsaAttentionKeyRopeInput => "dsa-attention-key-rope-input"
    DenseGateWeights => "dense-gate-weights"
    DenseUpWeights => "dense-up-weights"
    DenseDownWeights => "dense-down-weights"
    KdaSequenceOffsetsInput => "kda-sequence-offsets-input"
    KdaProjectedInput => "kda-projected-input"
    KdaShortConvolutionWeights => "kda-short-convolution-weights"
    KdaConvolutionInitialStateInput => "kda-convolution-initial-state-input"
    KdaConvolvedOutput => "kda-convolved-output"
    KdaConvolutionFinalStateOutput => "kda-convolution-final-state-output"
    KdaRecurrentQueryInput => "kda-recurrent-query-input"
    KdaRecurrentKeyInput => "kda-recurrent-key-input"
    KdaRecurrentValueInput => "kda-recurrent-value-input"
    KdaLogDecayInput => "kda-log-decay-input"
    KdaInputGateInput => "kda-input-gate-input"
    KdaRecurrentInitialStateInput => "kda-recurrent-initial-state-input"
    KdaRecurrentOutput => "kda-recurrent-output"
    KdaRecurrentFinalStateOutput => "kda-recurrent-final-state-output"
    KdaQueryProjectionWeights => "kda-query-projection-weights"
    KdaKeyProjectionWeights => "kda-key-projection-weights"
    KdaValueProjectionWeights => "kda-value-projection-weights"
    KdaForgetLowRankWeights => "kda-forget-low-rank-weights"
    KdaForgetExpansionWeights => "kda-forget-expansion-weights"
    KdaInputGateWeights => "kda-input-gate-weights"
    KdaProjectedQueryOutput => "kda-projected-query-output"
    KdaProjectedKeyOutput => "kda-projected-key-output"
    KdaProjectedValueOutput => "kda-projected-value-output"
    KdaForgetLowRankOutput => "kda-forget-low-rank-output"
    KdaRawForgetOutput => "kda-raw-forget-output"
    KdaBetaOutput => "kda-beta-output"
    KdaOutputGateLowRankWeights => "kda-output-gate-low-rank-weights"
    KdaOutputGateExpansionWeights => "kda-output-gate-expansion-weights"
    KdaOutputGateLowRankOutput => "kda-output-gate-low-rank-output"
    KdaOutputGateOutput => "kda-output-gate-output"
    KdaGatedNormRecurrentInput => "kda-gated-norm-recurrent-input"
    KdaGatedNormGateInput => "kda-gated-norm-gate-input"
    KdaGatedNormWeights => "kda-gated-norm-weights"
    KdaGatedNormOutput => "kda-gated-norm-output"
    KdaOutputProjectionInput => "kda-output-projection-input"
    KdaOutputProjectionWeights => "kda-output-projection-weights"
    KdaOutputProjectionOutput => "kda-output-projection-output"
    KdaDecayRawForgetInput => "kda-decay-raw-forget-input"
    KdaDecayBiasInput => "kda-decay-bias-input"
    KdaDecayLogRateInput => "kda-decay-log-rate-input"
    KdaDecayOutput => "kda-decay-output"
    MoeRouterFloat32Input => "moe-router-float32-input"
    MoeRouterBf16Weights => "moe-router-bf16-weights"
    MoeRouterFloat32LogitsOutput => "moe-router-float32-logits-output"
    MoeCorrectionBiasInput => "moe-correction-bias-input"
    MoeUncorrectedScoresOutput => "moe-uncorrected-scores-output"
    MoeChoiceScoresOutput => "moe-choice-scores-output"
    MoeSelectedIndicesOutput => "moe-selected-indices-output"
    MoeSelectedWeightsOutput => "moe-selected-weights-output"
    MoeFeedForwardInput => "moe-feed-forward-input"
    MoeExpertGateWeights => "moe-expert-gate-weights"
    MoeExpertUpWeights => "moe-expert-up-weights"
    MoeExpertDownWeights => "moe-expert-down-weights"
    MoeRoutedOutput => "moe-routed-output"
    MoeSharedGateWeights => "moe-shared-gate-weights"
    MoeSharedUpWeights => "moe-shared-up-weights"
    MoeSharedDownWeights => "moe-shared-down-weights"
    MoeSharedOutput => "moe-shared-output"
    MoeCombinedOutput => "moe-combined-output"
    DecoderResidualInput => "decoder-residual-input"
    DecoderResidualOutput => "decoder-residual-output"
    HyperPostControlInput => "hyper-post-control-input"
    HyperCombineControlInput => "hyper-combine-control-input"
    HyperResidualStreamsInput => "hyper-residual-streams-input"
    HyperSublayerOutputInput => "hyper-sublayer-output-input"
    HyperMixedStreamsOutput => "hyper-mixed-streams-output"
    HyperFunctionWeightsInput => "hyper-function-weights-input"
    HyperBaseInput => "hyper-base-input"
    HyperScaleInput => "hyper-scale-input"
    HyperInitialCombinationOutput => "hyper-initial-combination-output"
    HyperCollapsedOutput => "hyper-collapsed-output"
  }
}

///|
fn dense_projection_family_tag(
  family : GlmHybridDenseProjectionAotFamily,
) -> String {
  match family {
    AttentionOutputProjectionBf16 => "attention-output-projection-bf16"
    LanguageModelHeadBf16 => "language-model-head-bf16"
    DsaQueryAProjectionBf16 => "dsa-query-a-projection-bf16"
    DsaKeyValueAProjectionBf16 => "dsa-key-value-a-projection-bf16"
    DsaQueryBProjectionBf16 => "dsa-query-b-projection-bf16"
    DsaKeyValueBProjectionBf16 => "dsa-key-value-b-projection-bf16"
    KdaOutputProjectionBf16 => "kda-output-projection-bf16"
  }
}

///|
fn dense_projection_recipe(
  family : GlmHybridDenseProjectionAotFamily,
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  input_width : Int,
  output_width : Int,
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand],
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  let operand_text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(dense_projection_operand_tag(item.role()))
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment().to_string())
  }
  let stage_contract = match family {
    DsaQueryAProjectionBf16 =>
      "|stage=q-a-only|output=query-compressed-pre-rmsnorm" +
      "|excluded=q-a-rmsnorm-q-b-rope"
    DsaKeyValueAProjectionBf16 =>
      "|stage=kv-a-only|output=contiguous-compressed-kv-then-rope-key" +
      "|excluded=split-kv-rmsnorm-kv-b-rope"
    DsaQueryBProjectionBf16 =>
      "|stage=q-b-only|output=token-query-head-qk-component-view" +
      "|component-order=nope-then-rope|excluded=rope-application"
    DsaKeyValueBProjectionBf16 =>
      "|stage=kv-b-only|output=token-kv-head-key-nope-then-value" +
      "|value-subview=token-kv-head-value|excluded=kv-a-split-norm-and-rope-key"
    KdaOutputProjectionBf16 =>
      "|stage=kda-output-projection-only|input=flattened-token-head-dimension" +
      "|output=token-hidden|excluded=gated-rmsnorm-residual-cache"
    _ => ""
  }
  @utf8.encode(
    "lunaflux.glm-hybrid-dense-projection-cuda-aot.v1" +
    "|family=" +
    dense_projection_family_tag(family) +
    "|numerical=ordered-f32-accumulation-bf16-round-v1" +
    "|weights=row-major-output-input-bf16-no-bias" +
    stage_contract +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=" +
    input_width.to_string() +
    ":" +
    output_width.to_string() +
    ":" +
    maximum_tokens.to_string() +
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
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
fn lower_dense_projection_cuda_aot_candidate(
  family : GlmHybridDenseProjectionAotFamily,
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirement,
  requirement_ordinal~ : Int,
  input_width~ : Int,
  output_width~ : Int,
  output_role~ : @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole,
  input_role? : @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole = HiddenInput,
  weight_role? : @glm_hybrid_artifact_admission.GlmHybridLaunchOperandRole = ModelWeights,
  function_symbol~ : String,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseProjectionCudaCandidate raise GlmHybridCudaAotError {
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  if input_width <= 0 || output_width <= 0 {
    raise Invalid(Shape)
  }
  let input_elements = checked_mul(
    maximum_tokens.to_int64(),
    input_width.to_int64(),
  )
  let output_elements = checked_mul(
    maximum_tokens.to_int64(),
    output_width.to_int64(),
  )
  let weight_elements = checked_mul(
    input_width.to_int64(),
    output_width.to_int64(),
  )
  if output_elements > 9223372036854775552L {
    raise SizeOverflow
  }
  let grid = (output_elements + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(input_role, checked_mul(input_elements, 2L), 16L),
    dense_projection_operand(weight_role, checked_mul(weight_elements, 2L), 16L),
    dense_projection_operand(output_role, checked_mul(output_elements, 2L), 16L),
  ]
  let rendered = @decoder_projection_cuda_source.dense_projection_bf16_source(
    input_width~,
    output_width~,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    error => raise ProjectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = dense_projection_recipe(
    family, requirements, requirement_ordinal, input_width, output_width, maximum_tokens,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
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
    input_width,
    output_width,
    maximum_tokens,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: rendered.function_symbol(),
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}

///|
/// Lowers one exact terminal GLM language-model head through the shared,
/// family-neutral BF16 dense-projection renderer.
pub fn lower_language_model_head_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseProjectionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let input_width = geometry.hidden_size()
  let output_width = geometry.vocabulary_size()
  guard (requirement.operation(), requirement.capability())
    is (
      LanguageModelHead(operation_input, operation_output),
      Advanced(DenseProjectionAot),
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if operation_input != input_width || operation_output != output_width {
    raise Invalid(Shape)
  }
  lower_dense_projection_cuda_aot_candidate(
    LanguageModelHeadBf16,
    requirements,
    requirement,
    requirement_ordinal~,
    input_width~,
    output_width~,
    output_role=LogitsOutput,
    function_symbol="lunaflux_glm_hybrid_language_model_head_bf16_v1",
    maximum_tokens~,
    target~,
    compiler~,
    limits~,
  )
}
