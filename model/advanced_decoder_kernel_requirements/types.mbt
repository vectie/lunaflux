///|
/// Bounded startup failures. Operation ordinals are stable and contain no
/// artifact, prompt, or caller payload.
pub(all) suberror AdvancedDecoderKernelRequirementsError {
  InvalidLimit
  UnsupportedSchema
  InvalidSourceIdentity
  PlanExceedsLimits
  DuplicateSemanticOperation(Int32)
  RequirementsEncodingTooLarge
  UnsupportedGenericKernelOperation(Int32, AdvancedDecoderSemanticOperation)
} derive(Debug, Eq)

///|
/// Explicit startup ceilings. Derivation allocates and is never request-path
/// work.
pub struct AdvancedDecoderKernelRequirementLimits {
  priv maximum_layers : Int32
  priv maximum_auxiliary_stages : Int32
  priv maximum_requirements : Int32
} derive(Debug, Eq)

///|
pub fn AdvancedDecoderKernelRequirementLimits::new(
  maximum_layers~ : Int32,
  maximum_auxiliary_stages~ : Int32,
  maximum_requirements~ : Int32,
) -> Self raise AdvancedDecoderKernelRequirementsError {
  if maximum_layers <= 0 ||
    maximum_auxiliary_stages <= 0 ||
    maximum_requirements <= 0 {
    raise InvalidLimit
  }
  { maximum_layers, maximum_auxiliary_stages, maximum_requirements, }
}

///|
/// Exact family-neutral semantic work implied by an advanced decoder plan.
/// Geometry that changes AOT behavior remains in the operation value.
pub(all) enum AdvancedDecoderSemanticOperation {
  TokenEmbedding
  QueryLowRankAProjection(Int32)
  QueryLowRankBProjection(Int32, Int32, Int32)
  KeyValueProjection(Int32, Int32, Int32)
  QueryKeyRmsNorm(Double, Int32, Int32)
  RotaryPositionEncoding(Double, Double, Int32, Double, Double, Double)
  HyperConnectionBlockControlV1(Int32, Int32, Double, Double)
  HyperConnectionPreReduceV1(Int32)
  HyperConnectionPostCombineV1(Int32)
  HyperConnectionHeadReduceV1(Int32, Double, Double)
  SlidingWindowCausalAttention(Int32)
  LearnedCompressedIndex(Int32, Int32, Int32, Int32)
  IndexedCompressedAttention(Int32)
  DeterministicCompressedIndex(Int32)
  DeterministicWindowCompressedIndexJoinV1(Int32, Int32)
  NonOverlappingCompressedKeyValuePrepareV1(Int32, Int32)
  SelectedSparseAttentionV1(Int32, Int32)
  AttentionOutputInverseRotaryV1(Double, Int32, Double, Double, Double)
  AttentionOutputAProjectionV1(Int32, Int32, Int32, Int32)
  AttentionOutputBProjectionV1(Int32, Int32, Int32)
  ExpertScoreTransform(
    @advanced_decoder_execution_plan.ExpertScoreFunction,
    Int32
  )
  TokenHashRouting(Int32, Int32, Int32, Int32)
  NoAuxiliaryLossTopK(Int32, Int32)
  SelectedExpertWeightFinalize(Int32, Int32, Bool, Double)
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
  MultiTokenPredictionInputProjection
  DsparkTargetLayerRead(Int32)
  DsparkMainProjection
  DsparkNoiseEmbedding(Int32, Int32)
  AuxiliaryDecoderStage(Int32)
  AuxiliaryOutputHead
  DsparkMarkovAdjustment(Int32)
  DsparkConfidence
  FinalRmsNorm(Double)
  LanguageModelHead
} derive(Debug, Eq)

///|
/// Family-neutral semantic capability names for content-addressed AOT
/// artifacts. They are not vendor kernels or evidence of availability.
pub(all) enum AdvancedDecoderAotCapability {
  EmbeddingLookupAot
  LowRankAttentionProjectionAot
  QueryKeyRmsNormAot
  ScaledRotaryPositionAot
  HyperConnectionSinkhornAot
  HyperConnectionMixAot
  HyperConnectionBlockControlV1Aot
  HyperConnectionPreReduceV1Aot
  HyperConnectionPostCombineV1Aot
  HyperConnectionHeadReduceV1Aot
  QueryLowRankAProjectionAot
  QueryLowRankBProjectionAot
  DeterministicWindowCompressedIndexJoinV1Aot
  NonOverlappingCompressedKeyValuePrepareV1Aot
  SelectedSparseAttentionV1Aot
  AttentionOutputInverseRotaryV1Aot
  AttentionOutputAProjectionV1Aot
  AttentionOutputBProjectionV1Aot
  SlidingWindowCausalAttentionAot
  LearnedCompressedIndexAot
  IndexedCompressedAttentionAot
  DeterministicCompressedIndexAot
  ExpertScoreTransformAot
  TokenHashRoutingAot
  StableTopKRoutingAot
  SelectedExpertWeightFinalizeAot
  RoutedExpertFeedForwardAot
  SharedExpertFeedForwardAot
  RoutedSharedExpertCombineAot
  AuxiliaryProjectionAot
  AuxiliaryDecoderStageAot
  DsparkAot
  RmsNormAot
  DenseProjectionAot
} derive(Debug, Eq)

///|
/// One unique semantic operation paired with its required AOT capability.
pub struct AdvancedDecoderKernelRequirement {
  priv operation : AdvancedDecoderSemanticOperation
  priv capability : AdvancedDecoderAotCapability
} derive(Debug, Eq)

///|
/// Complete model-wide geometry needed to specialize shared embedding,
/// normalization, projection, attention, and logits kernels. Per-operation
/// geometry remains in `AdvancedDecoderSemanticOperation`.
pub struct AdvancedDecoderKernelGeometry {
  priv vocabulary_size : Int32
  priv hidden_size : Int32
  priv decoder_layer_count : Int32
  priv attention_head_count : Int32
  priv kv_head_count : Int32
  priv attention_head_dimension : Int32
  priv rope_head_dimension : Int32
  priv maximum_position_count : Int32
} derive(Debug, Eq)

///|
pub fn AdvancedDecoderKernelGeometry::vocabulary_size(self : Self) -> Int32 {
  self.vocabulary_size
}

///|
pub fn AdvancedDecoderKernelGeometry::hidden_size(self : Self) -> Int32 {
  self.hidden_size
}

///|
pub fn AdvancedDecoderKernelGeometry::decoder_layer_count(self : Self) -> Int32 {
  self.decoder_layer_count
}

///|
pub fn AdvancedDecoderKernelGeometry::attention_head_count(
  self : Self,
) -> Int32 {
  self.attention_head_count
}

///|
pub fn AdvancedDecoderKernelGeometry::kv_head_count(self : Self) -> Int32 {
  self.kv_head_count
}

///|
pub fn AdvancedDecoderKernelGeometry::attention_head_dimension(
  self : Self,
) -> Int32 {
  self.attention_head_dimension
}

///|
pub fn AdvancedDecoderKernelGeometry::rope_head_dimension(self : Self) -> Int32 {
  self.rope_head_dimension
}

///|
pub fn AdvancedDecoderKernelGeometry::maximum_position_count(
  self : Self,
) -> Int32 {
  self.maximum_position_count
}

///|
pub fn AdvancedDecoderKernelRequirement::operation(
  self : Self,
) -> AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderKernelRequirement::capability(
  self : Self,
) -> AdvancedDecoderAotCapability {
  self.capability
}

///|
/// Digest of the source identity plus the canonical ordered requirement set.
pub struct AdvancedDecoderKernelRequirementsDigest {
  priv hex : String
} derive(Debug, Eq)

///|
pub fn AdvancedDecoderKernelRequirementsDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
/// Immutable startup capability requirements derived from exactly one plan.
pub struct AdvancedDecoderKernelRequirements {
  priv source_identity : @spec.ModelIdentity
  priv source_schema_version : Int32
  priv geometry : AdvancedDecoderKernelGeometry
  priv digest : AdvancedDecoderKernelRequirementsDigest
  priv requirements : ReadOnlyArray[AdvancedDecoderKernelRequirement]
  priv capabilities : ReadOnlyArray[AdvancedDecoderAotCapability]
} derive(Debug, Eq)

///|
pub fn AdvancedDecoderKernelRequirements::geometry(
  self : Self,
) -> AdvancedDecoderKernelGeometry {
  self.geometry
}

///|
pub fn AdvancedDecoderKernelRequirements::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderKernelRequirements::source_schema_version(
  self : Self,
) -> Int32 {
  self.source_schema_version
}

///|
pub fn AdvancedDecoderKernelRequirements::digest(
  self : Self,
) -> AdvancedDecoderKernelRequirementsDigest {
  self.digest
}

///|
pub fn AdvancedDecoderKernelRequirements::requirements(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderKernelRequirement] {
  self.requirements
}

///|
pub fn AdvancedDecoderKernelRequirements::capabilities(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderAotCapability] {
  self.capabilities
}
