///|
/// Payload-safe rules for offline kernel-bundle admission.
pub(all) enum LunaKernelBundleRule {
  Limits
  Digest
  CompileReceipt
  Determinism
  ModuleBytes
  ModuleIdentity
  SourceIdentity
  RecipeIdentity
  OperationIdentity
  OperationKind
  EntryPoint
  LaunchDimensions
  OperandContract
  WorkspaceContract
  ModelIdentity
  DeviceTarget
  CatalogVersion
  OperationCoverage
  DuplicateOperation
  DuplicateModule
  TotalModuleBytes
  Manifest
} derive(Debug, Eq)

///|
/// Errors retain only bounded rules and integer operation locations. Compiler
/// paths, symbols, source, model payloads, and artifact bytes are excluded.
pub(all) suberror LunaKernelBundleError {
  Invalid(LunaKernelBundleRule)
  Unsupported(LunaKernelBundleRule)
  Mismatch(rule~ : LunaKernelBundleRule, operation_index~ : Int)
  LimitExceeded(rule~ : LunaKernelBundleRule, maximum~ : Int64)
} derive(Debug, Eq)
