///|
pub(all) suberror GlmHybridKernelRequirementsError {
  InvalidLimit
  UnsupportedSchema
  PlanExceedsLimits
  DuplicateSemanticOperation(Int32)
  RequirementsEncodingTooLarge
  UnsupportedGenericKernelOperation(Int32, GlmHybridSemanticOperation)
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirementLimits {
  priv maximum_layers : Int32
  priv maximum_vision_layers : Int32
  priv maximum_requirements : Int32
}

///|
pub fn GlmHybridKernelRequirementLimits::new(
  maximum_layers~ : Int32,
  maximum_vision_layers~ : Int32,
  maximum_requirements~ : Int32,
) -> Self raise GlmHybridKernelRequirementsError {
  if maximum_layers <= 0 ||
    maximum_vision_layers <= 0 ||
    maximum_requirements <= 0 {
    raise InvalidLimit
  }
  { maximum_layers, maximum_vision_layers, maximum_requirements, }
}

///|
pub(all) enum GlmHybridSemanticOperation {
  TokenEmbedding
  DecoderRmsNorm(Int32, Double)
  QueryLowRankProjection(Int32)
  KeyValueLowRankProjection(Int32, Int32, Int32, Int32)
  QueryKeyRmsNorm(Double)
  GlmRotaryPosition(Double?, Bool)
  DsaIndexCompute(Int32, Int32, Int32, Bool)
  DsaIndexPooling(Int32, Bool, Bool)
  DsaIndexReuse
  DsaSparseAttention(Int32, Int32, Int32, Int32)
  AttentionOutputProjection(Int32, Int32)
  KdaProjection(Int32, Int32)
  KdaShortConvolution(Int32, Int32)
  KdaRecurrentUpdate(Int32, Int32, Double)
  KdaOutputProjection(Int32, Int32)
  HyperConnectionFunction(Int32, Int32, Double, Double)
  HyperConnectionSinkhorn(Int32, Int32, Double)
  HyperConnectionMix(Int32)
  DenseSwiGlu(Int32, Int32)
  MoeRouterProjection(Int32, Int32, Bool)
  MoeSigmoidAndCorrectionBias(Int32)
  ExpertScoreTransform(@advanced_decoder_execution_plan.ExpertScoreFunction)
  MoeGroupedNoAuxiliaryLossTopK(Int32, Int32, Int32, Int32, Int32, Bool, Double)
  RoutedExpertFeedForward(
    Int32,
    Int32,
    @advanced_decoder_execution_plan.ExpertActivation
  )
  SharedExpertFeedForward(
    Int32,
    Int32,
    @advanced_decoder_execution_plan.ExpertActivation
  )
  RoutedSharedExpertCombine(Int32, Bool, Double)
  MoeDecoderResidualAdd(Int32)
  DsaKeyValueAttentionAssembly(Int32, Int32, Int32, Int32)
  MultiTokenPredictionInputProjection
  AuxiliaryDecoderStage(Int32)
  AuxiliaryOutputHead
  VisionPatchProjection(Int32, Int32, Int32, Int32)
  VisionRmsNorm(Int32)
  VisionQkvProjection(Int32, Int32)
  VisionAttention(Int32, Int32)
  VisionOutputProjection(Int32)
  VisionSwiGlu(Int32, Int32)
  VisionSpatialMerge(Int32, Int32, Int32)
  VisionDecoderProjection(Int32, Int32)
  VisionPostProjectionNorm(Int32)
  FinalRmsNorm(Int32, Double)
  LanguageModelHead(Int32, Int32)
  DsaQueryAProjection(Int32, Int32)
  DsaKeyValueAProjection(Int32, Int32, Int32)
  DsaQueryARmsNorm(Int32, Double)
  DsaQueryBProjection(Int32, Int32, Int32)
  KdaDecayControl(Int32, Int32, Double)
  KdaOutputGateProjection(Int32, Int32, Int32)
  KdaGatedRmsNorm(Int32, Int32, Double)
} derive(Debug, Eq)

///|
pub(all) enum GlmHybridAotCapability {
  Advanced(@advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability)
  DecoderRmsNormAot
  GlmRotaryPositionAot
  DsaIndexComputeAot
  DsaIndexPoolingAot
  DsaIndexReuseAot
  DsaSparseAttentionAot
  AttentionOutputProjectionAot
  KdaProjectionAot
  KdaShortConvolutionAot
  KdaRecurrentUpdateAot
  KdaOutputProjectionAot
  HyperConnectionFunctionAot
  DenseSwiGluAot
  MoeRouterProjectionAot
  MoeSigmoidAndCorrectionBiasAot
  MoeGroupedRoutingAot
  MoeDecoderResidualAddAot
  DsaKeyValueAttentionAssemblyAot
  VisionPatchProjectionAot
  VisionRmsNormAot
  VisionQkvProjectionAot
  VisionAttentionAot
  VisionOutputProjectionAot
  VisionSwiGluAot
  VisionSpatialMergeAot
  VisionDecoderProjectionAot
  VisionPostProjectionNormAot
  DsaQueryAProjectionAot
  DsaKeyValueAProjectionAot
  DsaQueryARmsNormAot
  DsaQueryBProjectionAot
  KdaDecayControlAot
  KdaOutputGateProjectionAot
  KdaGatedRmsNormAot
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirement {
  priv operation : GlmHybridSemanticOperation
  priv capability : GlmHybridAotCapability
} derive(Debug, Eq)

///|
/// Model-wide geometry shared by GLM decoder kernels. Specialized DSA, KDA,
/// MoE, mHC, and vision geometry remains in each semantic operation.
pub struct GlmHybridKernelGeometry {
  priv vocabulary_size : Int32
  priv hidden_size : Int32
  priv dense_intermediate_size : Int32
  priv decoder_layer_count : Int32
  priv context_length : Int32
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirementsDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirements {
  priv source_identity : @spec.ModelIdentity
  priv source_schema_version : Int32
  priv geometry : GlmHybridKernelGeometry
  priv digest : GlmHybridKernelRequirementsDigest
  priv requirements : ReadOnlyArray[GlmHybridKernelRequirement]
  priv capabilities : ReadOnlyArray[GlmHybridAotCapability]
} derive(Debug, Eq)
