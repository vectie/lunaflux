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

///|
pub extend AdvancedDecoderDimension with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderDimension with Eq::{not_equal, equal}

///|
pub extend AdvancedDecoderPlanError with @debug.Debug::{to_repr}

///|
pub extend AdvancedDecoderPlanError with Eq::{not_equal, equal}
