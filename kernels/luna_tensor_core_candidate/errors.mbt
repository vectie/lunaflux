///|
pub(all) enum LunaTileTensorCoreRule {
  Contract
  Target
  Shape
  Layout
  DType
  Alignment
  Epilogue
  Resource
  ProgramProof
  OracleIdentity
  ParallelIdentity
  SourceLimit
} derive(Debug, Eq)

///|
pub(all) suberror LunaTileTensorCoreCandidateError {
  LunaTileTensorCoreRejected(LunaTileTensorCoreRule)
} derive(Debug, Eq)

///|
pub extend LunaTileTensorCoreRule with @debug.Debug::{to_repr}

///|
pub extend LunaTileTensorCoreRule with Eq::{not_equal, equal}

///|
pub extend LunaTileTensorCoreCandidateError with @debug.Debug::{to_repr}

///|
pub extend LunaTileTensorCoreCandidateError with Eq::{not_equal, equal}
