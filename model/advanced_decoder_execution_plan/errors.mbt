///|
/// Dimension whose invalid value caused advanced decoder-plan admission to
/// fail. Error payloads contain no checkpoint paths or model content.
pub(all) enum AdvancedDecoderDimension {
  VocabularySize
  HiddenSize
  DecoderLayerCount
  AttentionHeadCount
  KeyValueHeadCount
  AttentionHeadDimension
  RopeHeadDimension
  QueryLowRank
  OutputLowRank
  OutputGroupCount
  MaximumPositionCount
} derive(Debug, Eq)

///|
/// Fail-closed construction errors for the architecture-neutral advanced
/// decoder execution plan.
pub(all) suberror AdvancedDecoderPlanError {
  InvalidSchemaVersion
  InvalidProfileIdentity
  InvalidDimension(AdvancedDecoderDimension)
  InvalidAttentionGeometry
  InvalidPositionEncoding
  InvalidHyperConnection
  InvalidExpertRouting
  InvalidLayerAttention
  InvalidLayerOrder
  LayerCountMismatch
  InvalidAuxiliaryPrediction
  CanonicalPlanTooLarge
} derive(Debug, Eq)
