///|
fn dsa_pooling_operand_tag(
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
fn dsa_pooling_index_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int) raise GlmHybridCudaAotError {
  let mut found : (Int, Int)? = None
  for requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexCompute(_, head_dimension, topk, _) {
      if found is Some(_) {
        raise Invalid(Requirement)
      }
      found = Some((head_dimension, topk))
    }
  }
  guard found is Some(result) else { raise Invalid(Requirement) }
  result
}

///|
fn dsa_pooling_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  head_dimension : Int,
  pool_size : Int,
  index_topk : Int,
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
    operand_text.write_string(dsa_pooling_operand_tag(item.role()))
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment().to_string())
  }
  @utf8.encode(
    "lunaflux.glm-hybrid-dsa-key-pooling-cuda-aot.v1" +
    "|family=dsa-key-pool-compression-bf16" +
    "|numerical=stable-f32-softmax-bf16-probability-ordered-f32-accumulation-v1" +
    "|semantics=first-valid-complete-all-valid-compact" +
    "|tail-policy=downstream-always-select-incomplete-visible-tail" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=" +
    head_dimension.to_string() +
    ":" +
    pool_size.to_string() +
    ":" +
    index_topk.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
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
/// Lowers the exact GLM-5.3-Flash DSA key-pool compression stage. The source
/// intentionally excludes scoring, top-k, and query-dependent tail selection.
pub fn lower_dsa_key_pooling_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDsaPoolingCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (DsaIndexPooling(pool_size, true, true), DsaIndexPoolingAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let (head_dimension, index_topk) = dsa_pooling_index_geometry(requirements)
  if pool_size != 4 || head_dimension != 128 || index_topk != 2048 {
    raise Invalid(Shape)
  }
  if maximum_tokens < pool_size ||
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
  let pool_capacity = maximum_tokens / pool_size
  let input_elements = checked_mul(
    maximum_tokens.to_int64(),
    head_dimension.to_int64(),
  )
  let output_elements = checked_mul(
    pool_capacity.to_int64(),
    head_dimension.to_int64(),
  )
  let ape_elements = checked_mul(
    pool_size.to_int64(),
    head_dimension.to_int64(),
  )
  let index_elements = checked_mul(
    pool_capacity.to_int64(),
    pool_size.to_int64(),
  )
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(
      DsaIndexKeysInput,
      checked_mul(input_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaIndexGateScoresInput,
      checked_mul(input_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaIndexValidityInput,
      checked_mul(maximum_tokens.to_int64(), 4L),
      4L,
    ),
    dense_projection_operand(ModelWeights, checked_mul(ape_elements, 2L), 16L),
    dense_projection_operand(
      DsaPooledKeysOutput,
      checked_mul(output_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaPooledRawIndicesOutput,
      checked_mul(index_elements, 4L),
      4L,
    ),
    dense_projection_operand(
      DsaPooledValidityOutput,
      checked_mul(pool_capacity.to_int64(), 4L),
      4L,
    ),
    dense_projection_operand(DsaPooledCountOutput, 4L, 4L),
  ]
  let rendered = @glm_hybrid_dsa_pooling_cuda_source.dsa_key_pool_compression_bf16_source(
    head_dimension~,
    pool_size~,
    maximum_tokens~,
    function_symbol="lunaflux_glm_hybrid_dsa_key_pool_bf16_v1",
  ) catch {
    error => raise DsaPoolingSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = dsa_pooling_recipe(
    requirements, requirement_ordinal, head_dimension, pool_size, index_topk, maximum_tokens,
    target, compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: DsaKeyPoolCompressionBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    head_dimension,
    pool_size,
    maximum_tokens,
    pool_capacity,
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
