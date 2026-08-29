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
