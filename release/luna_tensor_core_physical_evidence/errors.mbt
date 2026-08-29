///|
pub(all) enum LunaTensorCorePhysicalEvidenceRule {
  CanonicalDocument
  Digest
  Candidate
  Oracle
  ParallelFallback
  Target
  Device
  Compiler
  Sass
  Resource
  Numerical
  Sanitizer
  Seal
  Authority
} derive(Debug, Eq)

///|
pub(all) suberror LunaTensorCorePhysicalEvidenceError {
  LunaTensorCorePhysicalEvidenceInvalid(LunaTensorCorePhysicalEvidenceRule)
  LunaTensorCorePhysicalEvidenceMismatch(LunaTensorCorePhysicalEvidenceRule)
  LunaTensorCorePhysicalEvidenceLimitExceeded
} derive(Debug, Eq)
