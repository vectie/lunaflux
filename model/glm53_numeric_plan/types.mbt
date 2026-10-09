///|
/// Exact logical storage role retained alongside the generic numeric schema.
pub(all) enum Glm53LogicalTensorRole {
  BFloat16Parameter
  Float32RoutingCorrection
  Float32KdaStateParameter
  Float32HyperConnectionParameter
  Fp8Parameter
  Fp8BlockScaleMetadata
} derive(Debug, Eq)

///|
/// Stable graph location for one numeric operation requirement.
pub(all) enum Glm53NumericOperationScope {
  Global
  DecoderLayer(layer_index~ : Int)
  AuxiliaryLayer(stage_index~ : Int)
  VisionFrontend
  VisionLayer(layer_index~ : Int)
  VisionProjector
} derive(Debug, Eq)

///|
/// Closed semantic operation vocabulary for both official GLM-5.3 profiles.
pub(all) enum Glm53NumericOperationKind {
  TokenEmbedding
  LayerNorm
  HyperConnectionBf16Function
  HyperConnectionF32Control
  DsaLowRankProjection
  DsaIndexerProjection
  DsaIndexPooling
  DsaIndexSelection
  DsaIndexReuse
  DsaSparseAttention
  AttentionOutputProjection
  KdaProjection
  KdaShortConvolution
  KdaF32DecayControl
  KdaRecurrentUpdate
  DenseSwiGlu
  MoeRouterProjection
  MoeF32SigmoidAndCorrection
  MoeF32GroupedRouting
  RoutedExpertSwiGlu
  SharedExpertSwiGlu
  RoutedSharedExpertBf16Combine
  MtpFusionProjection
  SharedHeadNorm
  FinalNorm
  LanguageModelHead
  VisionPatchProjection
  VisionQkvProjection
  VisionAttention
  VisionOutputProjection
  VisionSwiGlu
  VisionSpatialMerge
  VisionProjector
  KdaOutputGateProjection
  KdaGatedRmsNorm
  KdaOutputProjection
} derive(Debug, Eq)

///|
pub(all) suberror Glm53NumericPlanError {
  UnsupportedFp8BlockScaleLayout(block_rows~ : Int, block_columns~ : Int)
  MixedDTypeMaterializationUnsupported
  ProfilePlanMismatch
  ManifestIdentityMismatch
  ManifestOrdinalMismatch(ordinal~ : Int)
  UnexpectedFloat32Tensor(ordinal~ : Int)
  NumericSchemaConstructionFailed
} derive(Debug, Eq)

///|
pub struct Glm53NumericTensorBinding {
  priv ordinal : Int
  priv role : Glm53LogicalTensorRole
}

///|
pub struct Glm53NumericOperationBinding {
  priv ordinal : Int
  priv scope : Glm53NumericOperationScope
  priv kind : Glm53NumericOperationKind
  priv execution : @numeric_contract.OperationExecutionContract
}

///|
/// Immutable logical numeric binding. It grants no physical authority.
pub struct Glm53NumericPlan {
  priv profile : @glm53_spec.Glm53ProfileId
  priv encoding : @glm53_weights.Glm53WeightEncoding
  priv execution_identity : @spec.ModelIdentity
  priv manifest_layout_digest : @spec.PlanDigest
  priv numeric_schema : @numeric_contract.ModelNumericSchema
  priv tensor_bindings : ReadOnlyArray[Glm53NumericTensorBinding]
  priv operation_bindings : ReadOnlyArray[Glm53NumericOperationBinding]
  priv binding_digest : @spec.PlanDigest
  priv bfloat16_tensor_count : Int
  priv float32_tensor_count : Int
}
