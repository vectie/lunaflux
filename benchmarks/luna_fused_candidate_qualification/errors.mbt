///|
pub(all) enum FusedCandidateBenchmarkStage {
  Protocol
  CandidateIdentity
  ShapeCoverage
  ObservationOrder
  WorkCounter
  DispatchCanary
  ResourceBalance
  DurationSummary
  WinPolicy
  EvidenceSize
} derive(Debug, Eq)

///|
pub(all) suberror FusedCandidateBenchmarkError {
  InvalidFusedCandidateBenchmark(FusedCandidateBenchmarkStage)
  MismatchedFusedCandidateBenchmark(FusedCandidateBenchmarkStage)
  OverflowedFusedCandidateBenchmark(FusedCandidateBenchmarkStage)
  FusedCandidateBenchmarkLimitExceeded(FusedCandidateBenchmarkStage)
} derive(Debug, Eq)
