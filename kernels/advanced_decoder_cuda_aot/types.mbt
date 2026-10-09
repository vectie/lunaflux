///|
pub(all) enum AdvancedDecoderFoundationAotFamily {
  TokenEmbeddingBf16
  QueryKeyRmsNormBf16
  FinalRmsNormBf16
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderFoundationNumericalContract {
  BitExactBf16Copy
  OrderedFloat32ReductionBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderProjectionAotFamily {
  LanguageModelHeadBf16
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderProjectionNumericalContract {
  OrderedF32AccumulationBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderRotaryAotFamily {
  CompressedScaledYarnRopeBf16
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderRotaryNumericalContract {
  AdjacentPairF32YarnBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderHyperConnectionAotFamily {
  BlockControlBf16F32V1
  PreReduceBf16V1
  PostCombineBf16V1
  HeadReduceBf16F32V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderHyperConnectionNumericalContract {
  OrderedF32ControlProjectionSinkhornV1
  OrderedF32PreReductionBf16RoundV1
  OrderedF32PostCombinationBf16RoundV1
  OrderedF32HeadControlReductionBf16RoundV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderRoutingAotFamily {
  ExpertScoreTransformF32
  TokenHashRoutingI32
  NoAuxiliaryLossTopKI32
  SelectedExpertWeightFinalizeF32
} derive(Debug, Eq)

///|
/// The source fixes the same finite-input piecewise sqrt-softplus formula as
/// the advanced-decoder scalar reference. Transcendental results remain bound
/// to the exact CUDA toolchain and require independent numeric qualification.
pub(all) enum AdvancedDecoderRoutingNumericalContract {
  FiniteInputStableSqrtSoftplusF32V1
  BitExactInt32TableLookupV1
  StableLowerExpertIdTopKV1
  OrderedF32NormalizeScaleV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderCompressedIndexAotFamily {
  CausalCompleteSlotsI32
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderCompressedIndexNumericalContract {
  BitExactCausalCompleteSlotsI32V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderWindowCompressedJoinAotFamily {
  DeterministicWindowCompressedJoinI32V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderWindowCompressedJoinNumericalContract {
  BitExactOfficialWindowThenCompressedI32V1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderSparseAttentionAotFamily {
  SharedKeyValueSelectedAttentionBf16
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderSparseAttentionNumericalContract {
  SerialOrderedF32SinkDenominatorBf16OutputV1
} derive(Debug, Eq)

///|
pub(all) enum AdvancedDecoderCudaAotInvalidKind {
  Limit
  Requirement
  Shape
  Target
  Compiler
  Encoding
} derive(Debug, Eq)

///|
pub(all) suberror AdvancedDecoderCudaAotError {
  Invalid(AdvancedDecoderCudaAotInvalidKind)
  UnsupportedRequirement(Int)
  SizeOverflow
  Operand(
    @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmissionError
  )
  Source(@decoder_foundation_cuda_source.DecoderFoundationCudaSourceError)
  ProjectionSource(
    @decoder_projection_cuda_source.DecoderProjectionCudaSourceError
  )
  TokenHashSource(
    @decoder_token_hash_cuda_source.DecoderTokenHashCudaSourceError
  )
  CompressedIndexSource(
    @decoder_compressed_index_cuda_source.DecoderCompressedIndexCudaSourceError
  )
  SparseAttentionSource(
    @decoder_selected_sparse_attention_cuda_source.DecoderSelectedSparseAttentionCudaSourceError
  )
  WindowCompressedJoinSource(
    @decoder_window_compressed_join_cuda_source.DecoderWindowCompressedJoinCudaSourceError
  )
  CompressedKvPrepareSource(
    @decoder_compressed_kv_prepare_cuda_source.DecoderCompressedKvPrepareCudaSourceError
  )
  RoutingSource(@decoder_routing_cuda_source.DecoderRoutingCudaSourceError)
  RotarySource(
    @decoder_scaled_rope_cuda_source.DecoderScaledRopeCudaSourceError
  )
  HyperConnectionSource(
    @decoder_hyper_connection_cuda_source.DecoderHyperConnectionCudaSourceError
  )
  BlockFp8Ue8M0Source(
    @block_fp8_ue8m0_projection_cuda_source.BlockFp8Ue8M0ProjectionCudaSourceError
  )
  Launch(@launch_contract.LaunchContractError)
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderCudaAotLimits {
  priv maximum_tokens : Int
  priv maximum_source_bytes : Int
  priv maximum_recipe_bytes : Int
}

///|
pub struct AdvancedDecoderCudaSourceDigest(String) derive(Debug, Eq, Hash)

///|
pub struct AdvancedDecoderCudaRecipeDigest(String) derive(Debug, Eq, Hash)

///|
/// Offline-only, non-bindable CUDA source candidate. It contains no module
/// digest, catalog family, loader, launch authority, or qualification claim.
pub struct AdvancedDecoderFoundationCudaCandidate {
  priv family : AdvancedDecoderFoundationAotFamily
  priv numerical_contract : AdvancedDecoderFoundationNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only, non-bindable dense-projection CUDA source candidate. It
/// carries no compiled module, loader, execution authority, or qualification
/// claim.
pub struct AdvancedDecoderProjectionCudaCandidate {
  priv family : AdvancedDecoderProjectionAotFamily
  priv numerical_contract : AdvancedDecoderProjectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv input_width : Int
  priv output_width : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only routing CUDA source candidate. It carries no compiled module,
/// loader, execution authority, or qualification claim.
pub struct AdvancedDecoderRoutingCudaCandidate {
  priv family : AdvancedDecoderRoutingAotFamily
  priv numerical_contract : AdvancedDecoderRoutingNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv routed_expert_count : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only deterministic compressed-index source candidate. It carries
/// no compiled module, KV ownership, loader, execution authority, or
/// qualification claim.
pub struct AdvancedDecoderCompressedIndexCudaCandidate {
  priv family : AdvancedDecoderCompressedIndexAotFamily
  priv numerical_contract : AdvancedDecoderCompressedIndexNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv compression_ratio : Int
  priv maximum_position_count : Int
  priv maximum_slots_per_row : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
pub struct AdvancedDecoderWindowCompressedJoinCudaCandidate {
  priv family : AdvancedDecoderWindowCompressedJoinAotFamily
  priv numerical_contract : AdvancedDecoderWindowCompressedJoinNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv window_size : Int
  priv compression_ratio : Int
  priv maximum_position_count : Int
  priv maximum_compressed_width : Int
  priv maximum_joined_width : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only source candidate for selected sparse attention over borrowed
/// shared K/V storage. It carries no index-construction, K/V ownership,
/// compiled module, loader, execution authority, or qualification claim.
pub struct AdvancedDecoderSparseAttentionCudaCandidate {
  priv family : AdvancedDecoderSparseAttentionAotFamily
  priv numerical_contract : AdvancedDecoderSparseAttentionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv maximum_batches : Int
  priv query_head_count : Int
  priv head_dimension : Int
  priv window_size : Int
  priv compression_ratio : Int
  priv selected_width : Int
  priv maximum_key_slots : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only source candidate for the compressed/scaled YaRN path. Buffers
/// contain RoPE-only slices; it conveys no compilation or execution authority.
pub struct AdvancedDecoderRotaryCudaCandidate {
  priv family : AdvancedDecoderRotaryAotFamily
  priv numerical_contract : AdvancedDecoderRotaryNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv query_head_count : Int
  priv key_value_head_count : Int
  priv rope_dimension : Int
  priv maximum_positions : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only source candidate for one exact mHC block-control phase. It
/// carries no compiled module, loader, launch authority, or qualification.
pub struct AdvancedDecoderHyperConnectionCudaCandidate {
  priv family : AdvancedDecoderHyperConnectionAotFamily
  priv numerical_contract : AdvancedDecoderHyperConnectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv stream_width : Int
  priv hidden_width : Int
  priv sinkhorn_iterations : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only source candidate for exact mHC pre-reduction. It carries no
/// compiled module, loader, launch authority, or qualification.
pub struct AdvancedDecoderHyperConnectionPreReduceCudaCandidate {
  priv family : AdvancedDecoderHyperConnectionAotFamily
  priv numerical_contract : AdvancedDecoderHyperConnectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv stream_width : Int
  priv hidden_width : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only source candidate for exact mHC post-combination. It carries no
/// compiled module, loader, launch authority, or qualification.
pub struct AdvancedDecoderHyperConnectionPostCombineCudaCandidate {
  priv family : AdvancedDecoderHyperConnectionAotFamily
  priv numerical_contract : AdvancedDecoderHyperConnectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv stream_width : Int
  priv hidden_width : Int
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}

///|
/// Offline-only source candidate for exact mHC head reduction. It carries no
/// compiled module, loader, launch authority, or qualification.
pub struct AdvancedDecoderHyperConnectionHeadReduceCudaCandidate {
  priv family : AdvancedDecoderHyperConnectionAotFamily
  priv numerical_contract : AdvancedDecoderHyperConnectionNumericalContract
  priv source_identity : @spec.ModelIdentity
  priv requirements_digest : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest
  priv requirement_ordinal : Int
  priv operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation
  priv capability : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability
  priv target : @catalog.DeviceTarget
  priv maximum_tokens : Int
  priv stream_width : Int
  priv hidden_width : Int
  priv control_epsilon : Double
  priv normalization_epsilon : Double
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ]
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : AdvancedDecoderCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : AdvancedDecoderCudaRecipeDigest
}
