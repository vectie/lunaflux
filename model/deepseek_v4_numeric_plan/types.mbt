///|
/// Closed logical storage semantics. Block shapes and canonical codebook
/// identity remain explicit because the generic numeric storage vocabulary
/// cannot faithfully express them.
pub(all) enum DeepSeekV4LogicalStorage {
  PlainBFloat16
  PlainFloat32
  PlainInt64
  Fp8E4M3Block128Parameter
  Ue8M0Block128ScaleMetadata
  PackedFp4E2M1Block32Parameter
  Ue8M0Block32ScaleMetadata
} derive(Debug, Eq)

///|
/// Coalesced logical tensor roles. Counts expand to the exact official tensor
/// cardinality while avoiding an unbounded 145,118-entry startup allocation.
pub(all) enum DeepSeekV4NumericTensorRole {
  GlobalBFloat16Parameter
  GlobalHyperConnectionFloat32
  AttentionBFloat16Parameter
  AttentionFloat32Control
  AttentionFp8Parameter
  AttentionScaleMetadata
  RouterBFloat16Weight
  RouterFloat32Bias
  TokenExpertHashInt64
  SharedExpertFp8Parameter
  SharedExpertScaleMetadata
  RoutedExpertFp8Parameter
  RoutedExpertFp4Parameter
  RoutedExpertBlock128ScaleMetadata
  RoutedExpertBlock32ScaleMetadata
  FfnNormBFloat16
  HyperConnectionFloat32
  CompressorBFloat16Parameter
  CompressorFloat32Control
  IndexerBFloat16Parameter
  IndexerFloat32Control
  IndexerFp8Parameter
  IndexerScaleMetadata
  MtpBFloat16Parameter
  MtpHyperConnectionFloat32
  MtpFp8Parameter
  MtpScaleMetadata
  DsparkBFloat16Parameter
  DsparkHyperConnectionFloat32
  DsparkFp8Parameter
  DsparkScaleMetadata
} derive(Debug, Eq)

///|
pub struct DeepSeekV4NumericTensorRun {
  priv role : DeepSeekV4NumericTensorRole
  priv storage : DeepSeekV4LogicalStorage
  priv tensor_count : Int
} derive(Debug, Eq)

///|
pub fn DeepSeekV4NumericTensorRun::role(
  self : Self,
) -> DeepSeekV4NumericTensorRole {
  self.role
}

///|
pub fn DeepSeekV4NumericTensorRun::storage(
  self : Self,
) -> DeepSeekV4LogicalStorage {
  self.storage
}

///|
pub fn DeepSeekV4NumericTensorRun::tensor_count(self : Self) -> Int {
  self.tensor_count
}

///|
pub(all) enum DeepSeekV4NumericOperationScope {
  Global
  DecoderLayer(Int)
  PredictionStage(Int)
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4HyperConnectionBlockSite {
  AttentionSite
  FeedForwardSite
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4NumericOperationKind {
  TokenEmbedding
  BlockRmsNorm
  HyperConnectionBlockControl(DeepSeekV4HyperConnectionBlockSite)
  HyperConnectionPreReduce(DeepSeekV4HyperConnectionBlockSite)
  HyperConnectionPostCombine(DeepSeekV4HyperConnectionBlockSite)
  HyperConnectionHeadReduce
  LowRankAttentionProjection
  QueryKeyRmsNorm
  ScaledRotaryPosition
  SlidingWindowAttention
  LearnedIndexSelection
  IndexedCompressedAttention
  DeterministicCompressedSelection
  DeterministicWindowCompressedIndexJoinV1
  NonOverlappingCompressedKeyValuePrepareV1
  SelectedSparseAttentionV1
  AttentionOutputInverseRotaryV1
  AttentionOutputAProjectionV1
  AttentionOutputBProjectionV1
  MoeRouterProjection
  ExpertScoreTransform
  TokenHashRouting
  StableTopKRouting
  RoutedExpertClampedSwiGlu
  SharedExpertClampedSwiGlu
  RoutedSharedExpertCombine
  MultiTokenPredictionInputProjection
  DsparkTargetLayerRead
  DsparkMainProjection
  DsparkNoiseEmbedding
  AuxiliaryOutputHead
  DsparkMarkovAdjustment
  DsparkConfidence
  FinalRmsNorm
  LanguageModelHead
} derive(Debug, Eq)

///|
/// A faithful generic execution contract or an explicit storage-semantics
/// rejection where generic compute vocabulary would imply nonexistent FP4
/// decoding authority.
pub(all) enum DeepSeekV4OperationExecution {
  NumericContract(@numeric_contract.OperationExecutionContract)
  RequiresCheckedInt64ToInt32TokenHash
  UnsupportedPackedFp4E2M1
} derive(Debug, Eq)

///|
pub struct DeepSeekV4NumericOperationBinding {
  priv ordinal : Int
  priv scope : DeepSeekV4NumericOperationScope
  priv kind : DeepSeekV4NumericOperationKind
  priv execution : DeepSeekV4OperationExecution
} derive(Debug, Eq)

///|
pub fn DeepSeekV4NumericOperationBinding::ordinal(self : Self) -> Int {
  self.ordinal
}

///|
pub fn DeepSeekV4NumericOperationBinding::scope(
  self : Self,
) -> DeepSeekV4NumericOperationScope {
  self.scope
}

///|
pub fn DeepSeekV4NumericOperationBinding::kind(
  self : Self,
) -> DeepSeekV4NumericOperationKind {
  self.kind
}

///|
pub fn DeepSeekV4NumericOperationBinding::execution(
  self : Self,
) -> DeepSeekV4OperationExecution {
  self.execution
}

///|
pub(all) enum DeepSeekV4Fp4CodebookBinding {
  CanonicalE2M1
} derive(Debug, Eq)

///|
pub(all) enum DeepSeekV4PhysicalReadiness {
  UnsupportedPhysicalMaterialization(
    ReadOnlyArray[@deepseek_v4_weights.DeepSeekV4WeightEncoding]
  )
} derive(Debug, Eq)

///|
pub(all) suberror DeepSeekV4NumericPlanError {
  ProfilePlanMismatch
  WeightManifestIdentityMismatch
  ManifestCardinalityMismatch(expected~ : Int, actual~ : Int)
  LogicalSchemaLimitExceeded
  PhysicalMaterializationUnsupported(
    ReadOnlyArray[@deepseek_v4_weights.DeepSeekV4WeightEncoding]
  )
} derive(Debug, Eq)

///|
/// Bounded immutable logical numeric plan. It grants no byte-layout,
/// dequantization, device, or kernel authority.
pub struct DeepSeekV4NumericPlan {
  priv profile : @deepseek_v4_spec.DeepSeekV4Profile
  priv identity : @spec.ModelIdentity
  priv logical_manifest_digest : @spec.PlanDigest
  priv numeric_schema_digest : @spec.PlanDigest
  priv binding_digest : @spec.PlanDigest
  priv expected_tensor_count : Int
  priv tensor_runs : ReadOnlyArray[DeepSeekV4NumericTensorRun]
  priv operations : ReadOnlyArray[DeepSeekV4NumericOperationBinding]
  priv fp4_codebook : DeepSeekV4Fp4CodebookBinding?
  priv physical_readiness : DeepSeekV4PhysicalReadiness
} derive(Debug, Eq)
