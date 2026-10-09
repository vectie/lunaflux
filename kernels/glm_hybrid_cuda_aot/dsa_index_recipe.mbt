///|
fn dsa_index_operand_tag(
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
fn dsa_index_family_tag(family : GlmHybridDsaIndexAotFamily) -> String {
  match family {
    RawTokenScoreStableTopKBf16 => "raw-token-score-stable-topk-bf16"
    PooledScoreStableTopKWithTailBf16 =>
      "pooled-score-stable-topk-with-tail-bf16"
    SelectedIndexReuseI32 => "selected-index-reuse-i32"
  }
}

///|
fn dsa_index_numerical_tag(
  contract : @glm_hybrid_dsa_index_cuda_source.GlmHybridDsaIndexNumericalContract,
) -> String {
  match contract {
    SerialOrderedFloat32ReluWeightedScoreStableTopKV1 =>
      "serial-ordered-f32-relu-weighted-score-stable-topk-v1"
    BitExactI32Copy => "bit-exact-i32-copy"
  }
}

///|
fn dsa_index_recipe(
  family : GlmHybridDsaIndexAotFamily,
  numerical_contract : @glm_hybrid_dsa_index_cuda_source.GlmHybridDsaIndexNumericalContract,
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  head_count : Int,
  head_dimension : Int,
  top_k : Int,
  pool_size : Int,
  maximum_tokens : Int,
  candidate_capacity : Int,
  output_width : Int,
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
    operand_text.write_string(dsa_index_operand_tag(item.role()))
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment().to_string())
  }
  let stage_contract = match family {
    SelectedIndexReuseI32 =>
      "|upstream=previous-selected-indices|copy=bit-exact-counts-bounded"
    _ =>
      "|upstream=projected-queries-head-weights-and-candidate-keys" +
      "|key-validity=single-contiguous-span" +
      "|batch-contract=single-sequence-counts-2-equals-1" +
      "|tie-break=lower-candidate-ordinal" +
      "|tail-offset=selected-pool-count-times-pool-size-before-padding"
  }
  @utf8.encode(
    "lunaflux.glm-hybrid-dsa-index-cuda-aot.v1" +
    "|family=" +
    dsa_index_family_tag(family) +
    "|numerical=" +
    dsa_index_numerical_tag(numerical_contract) +
    "|qualification=serial-correctness-only-not-throughput-qualified" +
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
    head_count.to_string() +
    ":" +
    head_dimension.to_string() +
    ":" +
    top_k.to_string() +
    ":" +
    pool_size.to_string() +
    ":" +
    maximum_tokens.to_string() +
    ":" +
    candidate_capacity.to_string() +
    ":" +
    output_width.to_string() +
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
