///|
pub(all) enum ReadOnlyProductionRule {
  Qualification
  PhysicalPolicy
  ExternalApproval
  RuntimeDevice
  Candidate
  CompiledArtifact
  PhysicalEvidence
  Target
  Profile
  Layout
  Operation
  Activation
  KernelAbi
  Launch
  Canary
  Catalog
} derive(Debug, Eq)

///|
pub(all) suberror ReadOnlyProductionError {
  InvalidReadOnlyProduction(ReadOnlyProductionRule)
  MismatchedReadOnlyProduction(ReadOnlyProductionRule)
} derive(Debug, Eq)
