///|
pub(all) enum Fp8LaunchAbiRule {
  InvalidShape
  ShapeOverflow
  InvalidStorageIdentity
  InvalidStorageRegion
  RegionOverflow
  RegionOutOfBounds
  RegionAlignment
  RegionStorageMismatch
  RegionAlias
  UnsupportedScaleMode
  ActivationScalePolicy
  ActivationScaleStage
  OperationKind
  ModelIdentity
  NumericSchema
  ManifestScope
  ManifestEntry
  LaunchContract
  ExecutionDigest
  KernelIdentity
  WorkspaceContract
  BindingMismatch
  StaleLease
  EpochExhausted
  InvalidTransition
} derive(Debug, Eq)

///|
pub(all) suberror Fp8LaunchAbiError {
  InvalidFp8LaunchAbi(Fp8LaunchAbiRule)
} derive(Debug, Eq)
