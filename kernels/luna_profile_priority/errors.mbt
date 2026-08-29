///|
pub(all) enum LunaKernelProfileRule {
  Digest
  Limit
  Observation
  ObservationOrder
  ExecutionMode
  Operation
  Shape
  Target
  Timing
  Total
  CanonicalBytes
  Contract
  PageTable
  Row
  RowOrder
  Counter
  CounterOrder
  LaunchIdentity
} derive(Debug, Eq)

///|
pub(all) suberror LunaKernelProfileError {
  LunaKernelProfileInvalid(LunaKernelProfileRule)
  LunaKernelProfileMismatch(LunaKernelProfileRule)
  LunaKernelProfileUnsupported(LunaKernelProfileRule)
} derive(Debug, Eq)
