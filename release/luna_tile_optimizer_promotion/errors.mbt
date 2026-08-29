///|
pub(all) enum LunaTileOptimizerPromotionRule {
  Digest
  Candidate
  Fallback
  Target
  PhysicalEvidence
  Benchmark
  NonWin
  Artifact
  ExternalApproval
  Deployment
  Replay
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror LunaTileOptimizerPromotionError {
  LunaTileOptimizerPromotionInvalid(LunaTileOptimizerPromotionRule)
  LunaTileOptimizerPromotionMismatch(LunaTileOptimizerPromotionRule)
  LunaTileOptimizerPromotionUnsupported(LunaTileOptimizerPromotionRule)
} derive(Debug, Eq)
