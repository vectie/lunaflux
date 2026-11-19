///|
/// Dimension whose value violated the dense Llama model contract.
pub(all) enum ModelDimension {
  BatchRows
  Vocabulary
  Hidden
  Intermediate
  LayerCount
  AttentionHeadCount
  KvHeadCount
  ContextLength
} derive(Debug, Eq)

///|
/// Relationship between otherwise positive dimensions that was invalid.
pub(all) enum HeadRelationship {
  HiddenNotDivisibleByAttentionHeads
  AttentionHeadsNotDivisibleByKvHeads
  KvHeadsExceedAttentionHeads
  HeadDimensionMismatch
} derive(Debug, Eq)

///|
/// Bounded, payload-safe model metadata validation failures.
///
/// No constructor carries paths, raw configuration text, or untrusted values.
pub(all) suberror ModelSpecError {
  InvalidDimension(ModelDimension)
  InvalidHeadRelationship(HeadRelationship)
  InvalidRopeTheta
  InvalidRmsNormEpsilon
  UnsupportedWeightDType
  UnsupportedKvDType
  UnknownDType
  InvalidDigest
} derive(Debug, Eq)

///|
pub extend ModelDimension with @debug.Debug::{to_repr}

///|
pub extend ModelDimension with Eq::{not_equal, equal}

///|
pub extend HeadRelationship with @debug.Debug::{to_repr}

///|
pub extend HeadRelationship with Eq::{not_equal, equal}

///|
pub extend ModelSpecError with @debug.Debug::{to_repr}

///|
pub extend ModelSpecError with Eq::{not_equal, equal}
