///|
/// Payload-safe validation classes for offline full-graph manifest production.
pub(all) enum FullGraphManifestRule {
  Limits
  ContractScope
  CatalogVersion
  ModelIdentity
  DeviceTarget
  Profile
  OperationCoverage
  OperationOrder
  EntryPoint
  LaunchDimensions
  OperandContract
  WorkspaceContract
  ModulePath
  FunctionSymbol
  ModuleOrdering
  ManifestBytes
} derive(Debug, Eq)

///|
/// Offline production failures never retain a path, symbol, model payload, or
/// generated document fragment.
pub(all) suberror FullGraphManifestError {
  Invalid(FullGraphManifestRule)
  Mismatch(rule~ : FullGraphManifestRule, operation_index~ : Int)
  LimitExceeded(rule~ : FullGraphManifestRule, maximum~ : Int64)
} derive(Debug, Eq)
