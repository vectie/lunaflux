///|
pub(all) enum FusedPhysicalEvidenceRule {
  CanonicalDocument
  Digest
  Family
  Candidate
  Compiler
  Target
  Geometry
  Artifact
  Referee
  Numerical
  DispatchCanary
  Sanitizer
  ResourceBalance
  Seal
  Benchmark
  ExternalApproval
  Authority
} derive(Debug, Eq)

///|
pub(all) suberror FusedPhysicalEvidenceError {
  FusedPhysicalEvidenceInvalid(FusedPhysicalEvidenceRule)
  FusedPhysicalEvidenceMismatch(FusedPhysicalEvidenceRule)
  FusedPhysicalEvidenceLimitExceeded
} derive(Debug, Eq)
