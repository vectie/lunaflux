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

///|
pub extend LunaTensorCorePhysicalEvidenceRule with @debug.Debug::{to_repr}

///|
pub extend LunaTensorCorePhysicalEvidenceRule with Eq::{not_equal, equal}

///|
pub extend LunaTensorCorePhysicalEvidenceError with @debug.Debug::{to_repr}

///|
pub extend LunaTensorCorePhysicalEvidenceError with Eq::{not_equal, equal}
