///|
pub(all) enum KernelArtifactLimit {
  ModuleCount
  EntryPointCount
  ModuleBytes
  TotalModuleBytes
  FunctionSymbolBytes
} derive(Debug, Eq)

///|
pub(all) enum FunctionSymbolRule {
  Empty
  TooLong
  NonAscii
  InvalidInitial
  InvalidContinuation
} derive(Debug, Eq)

///|
pub(all) suberror KernelArtifactError {
  CatalogVersionMismatch(
    expected~ : @catalog.CatalogVersion,
    actual~ : @catalog.CatalogVersion
  )
  InvalidLimit(KernelArtifactLimit)
  ModuleCountExceeded(maximum~ : Int)
  EntryPointCountExceeded(maximum~ : Int)
  EmptyModule(@catalog.AotArtifactDigest)
  ModuleTooLarge(digest~ : @catalog.AotArtifactDigest, maximum~ : Int64)
  TotalModuleBytesExceeded(maximum~ : Int64)
  InvalidFunctionSymbol(
    entry_point~ : @catalog.AotKernelEntryPoint,
    rule~ : FunctionSymbolRule
  )
  DigestMismatch(@catalog.AotArtifactDigest)
  DuplicateModule(@catalog.AotArtifactDigest)
  MissingModule(@catalog.AotArtifactDigest)
  UnreferencedModule(@catalog.AotArtifactDigest)
  DuplicateEntryPoint(@catalog.AotKernelEntryPoint)
  MissingEntryPoint(@catalog.AotKernelEntryPoint)
  UnreferencedEntryPoint(@catalog.AotKernelEntryPoint)
} derive(Debug, Eq)
