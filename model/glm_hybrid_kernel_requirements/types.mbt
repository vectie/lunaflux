///|
pub(all) suberror GlmHybridKernelRequirementsError {
  InvalidLimit
  UnsupportedSchema
  PlanExceedsLimits
  DuplicateSemanticOperation(Int)
  RequirementsEncodingTooLarge
  UnsupportedGenericKernelOperation(Int, GlmHybridSemanticOperation)
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirementLimits {
  priv maximum_layers : Int
  priv maximum_vision_layers : Int
  priv maximum_requirements : Int
}

///|
pub fn GlmHybridKernelRequirementLimits::new(
  maximum_layers~ : Int,
  maximum_vision_layers~ : Int,
  maximum_requirements~ : Int,
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
  DecoderRmsNorm(Int, Double)
  QueryLowRankProjection(Int)
  KeyValueLowRankProjection(Int, Int, Int, Int)
  QueryKeyRmsNorm(Double)
  GlmRotaryPosition(Double?, Bool)
  DsaIndexCompute(Int, Int, Int, Bool)
  DsaIndexPooling(Int, Bool, Bool)
  DsaIndexReuse
  DsaSparseAttention(Int, Int, Int, Int)
  AttentionOutputProjection(Int, Int)
  KdaProjection(Int, Int)
  KdaShortConvolution(Int, Int)
  KdaRecurrentUpdate(Int, Int, Double)
  KdaOutputProjection(Int, Int)
  HyperConnectionFunction(Int, Int, Double, Double)
  HyperConnectionSinkhorn(Int, Int, Double)
  HyperConnectionMix(Int)
  DenseSwiGlu(Int, Int)
  MoeRouterProjection(Int, Int, Bool)
  MoeSigmoidAndCorrectionBias(Int)
  ExpertScoreTransform(@advanced_decoder_execution_plan.ExpertScoreFunction)
  MoeGroupedNoAuxiliaryLossTopK(Int, Int, Int, Int, Int, Bool, Double)
  RoutedExpertFeedForward(
    Int,
    Int,
    @advanced_decoder_execution_plan.ExpertActivation
  )
  SharedExpertFeedForward(
    Int,
    Int,
    @advanced_decoder_execution_plan.ExpertActivation
  )
  RoutedSharedExpertCombine(Int, Bool, Double)
  MoeDecoderResidualAdd(Int)
  DsaKeyValueAttentionAssembly(Int, Int, Int, Int)
  MultiTokenPredictionInputProjection
  AuxiliaryDecoderStage(Int)
  AuxiliaryOutputHead
  VisionPatchProjection(Int, Int, Int, Int)
  VisionRmsNorm(Int)
  VisionQkvProjection(Int, Int)
  VisionAttention(Int, Int)
  VisionOutputProjection(Int)
  VisionSwiGlu(Int, Int)
  VisionSpatialMerge(Int, Int, Int)
  VisionDecoderProjection(Int, Int)
  VisionPostProjectionNorm(Int)
  FinalRmsNorm(Int, Double)
  LanguageModelHead(Int, Int)
  DsaQueryAProjection(Int, Int)
  DsaKeyValueAProjection(Int, Int, Int)
  DsaQueryARmsNorm(Int, Double)
  DsaQueryBProjection(Int, Int, Int)
  KdaDecayControl(Int, Int, Double)
  KdaOutputGateProjection(Int, Int, Int)
  KdaGatedRmsNorm(Int, Int, Double)
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
  priv vocabulary_size : Int
  priv hidden_size : Int
  priv dense_intermediate_size : Int
  priv decoder_layer_count : Int
  priv context_length : Int
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirementsDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub struct GlmHybridKernelRequirements {
  priv source_identity : @spec.ModelIdentity
  priv source_schema_version : Int
  priv geometry : GlmHybridKernelGeometry
  priv digest : GlmHybridKernelRequirementsDigest
  priv requirements : ReadOnlyArray[GlmHybridKernelRequirement]
  priv capabilities : ReadOnlyArray[GlmHybridAotCapability]
} derive(Debug, Eq)
