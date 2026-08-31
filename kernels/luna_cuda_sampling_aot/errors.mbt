///|
/// Exact invariant audited at the Phase-5 sampling ownership boundary.
pub(all) enum SamplingBoundaryRule {
  FinalOutputProducer
  FinalOutputKind
  FinalOutputShape
  FinalOutputDtype
  VocabularySize
  MaximumRows
  ArtifactLimits
  ArtifactTarget
  ArtifactSource
  ArtifactRecipe
  ArtifactModule
  ArtifactDigest
  ArtifactReceipt
  ArtifactDeterminism
  DeviceSamplingPlan
  DeviceSamplingParameters
} derive(Debug, Eq)

///|
/// Payload-free failure from sampling-boundary admission.
pub(all) suberror SamplingBoundaryError {
  Invalid(SamplingBoundaryRule)
  Unsupported(SamplingBoundaryRule)
  SizeOverflow(SamplingBoundaryRule)
  Mismatch(SamplingBoundaryRule)
  LimitExceeded(SamplingBoundaryRule)
} derive(Debug, Eq)
