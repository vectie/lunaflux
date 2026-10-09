///|
pub(all) enum GlmHybridFoundationAotFamily {
  TokenEmbeddingBf16
  DecoderRmsNormBf16
  FinalRmsNormBf16
  DsaQueryARmsNormBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridDenseProjectionAotFamily {
  AttentionOutputProjectionBf16
  LanguageModelHeadBf16
  DsaQueryAProjectionBf16
  DsaKeyValueAProjectionBf16
  DsaQueryBProjectionBf16
  DsaKeyValueBProjectionBf16
  KdaOutputProjectionBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridDsaPoolingAotFamily {
  DsaKeyPoolCompressionBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridDsaIndexAotFamily {
  RawTokenScoreStableTopKBf16
  PooledScoreStableTopKWithTailBf16
  SelectedIndexReuseI32
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridSparseAttentionAotFamily {
  SelectedIndexSparseAttentionBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridKvSplitNormAotFamily {
  DsaKeyValuePrefixRmsNormBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridRotaryAotFamily {
  DsaInterleavedQuerySharedKeyRopeBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridDenseSwiGluAotFamily {
  SeparateGateUpDownWeightsBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridKdaShortConvolutionAotFamily {
  ReusableDepthwiseCausalSiluBf16
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridKdaRecurrentAotFamily {
  SerialNormalizedRecurrentDeltaBf16F32
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridFoundationOperandRole {
  StepCountsInput
  TokenIdsInput
  HiddenInput
  ModelWeights
  HiddenOutput
  DsaQueryCompressedInput
  DsaQueryNormalizedOutput
} derive(Debug, Eq)

///|
pub struct GlmHybridFoundationOperand {
  priv role : GlmHybridFoundationOperandRole
  priv byte_count : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridCudaAotInvalidKind {
  Limit
  Requirement
  Shape
  Target
  Compiler
  Encoding
} derive(Debug, Eq)

///|
pub(all) suberror GlmHybridCudaAotError {
  Invalid(GlmHybridCudaAotInvalidKind)
  UnsupportedRequirement(Int)
  SizeOverflow
  Source(@decoder_foundation_cuda_source.DecoderFoundationCudaSourceError)
  ProjectionSource(
    @decoder_projection_cuda_source.DecoderProjectionCudaSourceError
  )
  DsaPoolingSource(
    @glm_hybrid_dsa_pooling_cuda_source.GlmHybridDsaPoolingCudaSourceError
  )
  DsaIndexSource(
    @glm_hybrid_dsa_index_cuda_source.GlmHybridDsaIndexCudaSourceError
  )
  SparseAttentionSource(
    @glm_hybrid_sparse_attention_cuda_source.GlmHybridSparseAttentionCudaSourceError
  )
  PrefixNormSource(@prefix_rms_norm_cuda_source.PrefixRmsNormCudaSourceError)
  InterleavedRopeSource(
    @decoder_interleaved_rope_cuda_source.DecoderInterleavedRopeCudaSourceError
  )
  GatedMlpSource(@gated_mlp_cuda_source.GatedMlpCudaSourceError)
  GatedRmsNormSource(@gated_rms_norm_cuda_source.GatedRmsNormCudaSourceError)
  DecayControlSource(@decay_control_cuda_source.DecayControlCudaSourceError)
  RouterProjectionSource(
    @router_projection_cuda_source.RouterProjectionCudaSourceError
  )
  SigmoidCorrectionSource(
    @sigmoid_correction_cuda_source.SigmoidCorrectionCudaSourceError
  )
  GroupedRoutingSource(
    @grouped_routing_cuda_source.GroupedRoutingCudaSourceError
  )
  SelectedExpertMlpSource(
    @selected_expert_mlp_cuda_source.SelectedExpertMlpCudaSourceError
  )
  Bf16BinarySource(@bf16_binary_cuda_source.Bf16BinaryCudaSourceError)
  HyperConnectionMixSource(
    @hyper_connection_mix_cuda_source.HyperConnectionMixCudaSourceError
  )
  HyperConnectionControlSource(
    @decoder_hyper_connection_cuda_source.DecoderHyperConnectionCudaSourceError
  )
  KvAssemblySource(
    @decoder_kv_assembly_cuda_source.DecoderKvAssemblyCudaSourceError
  )
  CausalShortConvolutionSource(
    @causal_short_convolution_cuda_source.CausalShortConvolutionCudaSourceError
  )
  RecurrentDeltaSource(
    @recurrent_delta_cuda_source.RecurrentDeltaCudaSourceError
  )
  RecurrentAttentionProjectionSource(
    @recurrent_attention_projection_cuda_source.RecurrentAttentionProjectionCudaSourceError
  )
  TwoStageProjectionSource(
    @two_stage_projection_cuda_source.TwoStageProjectionCudaSourceError
  )
  Operand(@glm_hybrid_artifact_admission.GlmHybridArtifactAdmissionError)
  Launch(@launch_contract.LaunchContractError)
} derive(Debug, Eq)

///|
pub struct GlmHybridCudaAotLimits {
  priv maximum_tokens : Int
  priv maximum_source_bytes : Int
  priv maximum_recipe_bytes : Int
}

///|
pub struct GlmHybridCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub struct GlmHybridCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
/// Offline-only GLM adapter over family-neutral decoder foundation source.
/// It is deliberately non-bindable until compiled artifacts are qualified.
pub struct GlmHybridFoundationCudaCandidate {
  priv family : GlmHybridFoundationAotFamily
  priv numerical_contract : @decoder_foundation_cuda_source.DecoderFoundationNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[GlmHybridFoundationOperand]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable BF16 dense-projection evidence. It binds family-neutral source
/// to one exact GLM semantic operation and geometry but grants no compiled-
/// artifact, device, numeric-manifest, or qualification authority.
pub struct GlmHybridDenseProjectionCudaCandidate {
  priv family : GlmHybridDenseProjectionAotFamily
  priv numerical_contract : @decoder_projection_cuda_source.DecoderProjectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv input_width : Int
  priv output_width : Int
  priv maximum_tokens : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable evidence for the standalone DSA key-pool compression stage.
/// Candidate scoring, top-k, and query-dependent tail selection are excluded.
pub struct GlmHybridDsaPoolingCudaCandidate {
  priv family : GlmHybridDsaPoolingAotFamily
  priv numerical_contract : @glm_hybrid_dsa_pooling_cuda_source.GlmHybridDsaPoolingNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv head_dimension : Int
  priv pool_size : Int
  priv maximum_tokens : Int
  priv pool_capacity : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable projected DSA index scoring/selection or exact reuse evidence.
/// Upstream projection and pooling remain separate execution boundaries.
pub struct GlmHybridDsaIndexCudaCandidate {
  priv family : GlmHybridDsaIndexAotFamily
  priv numerical_contract : @glm_hybrid_dsa_index_cuda_source.GlmHybridDsaIndexNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv head_count : Int
  priv head_dimension : Int
  priv top_k : Int
  priv pool_size : Int
  priv maximum_tokens : Int
  priv candidate_capacity : Int
  priv output_width : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable serial correctness evidence over already-projected Q/K/V.
pub struct GlmHybridSparseAttentionCudaCandidate {
  priv family : GlmHybridSparseAttentionAotFamily
  priv numerical_contract : @glm_hybrid_sparse_attention_cuda_source.GlmHybridSparseAttentionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv query_head_count : Int
  priv key_value_head_count : Int
  priv query_key_dimension : Int
  priv value_dimension : Int
  priv top_k : Int
  priv selected_width : Int
  priv maximum_tokens : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
pub struct GlmHybridKvSplitNormCudaCandidate {
  priv family : GlmHybridKvSplitNormAotFamily
  priv numerical_contract : @prefix_rms_norm_cuda_source.PrefixRmsNormNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv prefix_width : Int
  priv suffix_width : Int
  priv maximum_tokens : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable evidence for Full GLM interleaved query/shared-key RoPE.
pub struct GlmHybridRotaryCudaCandidate {
  priv family : GlmHybridRotaryAotFamily
  priv numerical_contract : @decoder_interleaved_rope_cuda_source.DecoderInterleavedRopeNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv query_head_count : Int
  priv query_head_dimension : Int
  priv rope_dimension : Int
  priv maximum_tokens : Int
  priv maximum_positions : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable correctness candidate for one counts-bounded dense SwiGLU.
pub struct GlmHybridDenseSwiGluCudaCandidate {
  priv family : GlmHybridDenseSwiGluAotFamily
  priv numerical_contract : @gated_mlp_cuda_source.GatedMlpNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv hidden_size : Int
  priv intermediate_size : Int
  priv maximum_tokens : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable correctness candidate for one Q, K, or V short-convolution
/// invocation. The same artifact is reusable for all three projected streams.
pub struct GlmHybridKdaShortConvolutionCudaCandidate {
  priv family : GlmHybridKdaShortConvolutionAotFamily
  priv numerical_contract : @causal_short_convolution_cuda_source.CausalShortConvolutionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv channel_count : Int
  priv kernel_size : Int
  priv maximum_tokens : Int
  priv maximum_sequences : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}

///|
/// Non-bindable serial correctness candidate for the normalized KDA recurrent
/// delta state transition. Gate projection and persistent cache ownership are
/// outside the candidate.
pub struct GlmHybridKdaRecurrentCudaCandidate {
  priv family : GlmHybridKdaRecurrentAotFamily
  priv numerical_contract : @recurrent_delta_cuda_source.RecurrentDeltaNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation
  priv capability : @glm_hybrid_kernel_requirements.GlmHybridAotCapability
  priv head_count : Int
  priv head_dimension : Int
  priv normalization_epsilon : Double
  priv maximum_tokens : Int
  priv maximum_sequences : Int
  priv target : @catalog.DeviceTarget
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @glm_hybrid_artifact_admission.GlmHybridLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : GlmHybridCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : GlmHybridCudaRecipeDigest
}
