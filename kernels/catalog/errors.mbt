///|
/// Bounded validation reasons for catalog data supplied at startup.
pub(all) enum CatalogValidationRule {
  NonPositiveVersion
  UnsupportedCatalogVersion
  UnsupportedSemanticVersion
  EmptyCatalog
  CatalogTooLarge
  InvalidComputeCapability
  InvalidArtifactDigest
  InvalidWorkspaceBytes
  InvalidWorkspaceAlignment
  InvalidVendorImplementationKind
} derive(Debug, Eq)

///|
/// Startup kernel selection failures. Operation IDs are semantic plan
/// positions and do not disclose device handles, paths, or vendor diagnostics.
pub(all) suberror CatalogError {
  InvalidCatalog(CatalogValidationRule)
  MissingKernel(@plan.OperationId)
  IncompatibleTarget(@plan.OperationId)
  IncompatibleImplementation(@plan.OperationId)
  AmbiguousKernel(@plan.OperationId)
  WorkspaceSizeOverflow(@plan.OperationId)
  WorkspaceRequirementMismatch(@plan.OperationId)
  WorkspaceLimitExceeded(@plan.OperationId)
} derive(Debug, Eq)
