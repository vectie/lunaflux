///|
/// Resource envelopes validated before an artifact bundle retains module data.
pub(all) enum KernelArtifactLimit {
  ArtifactCount
  ModuleBytes
  TotalModuleBytes
  FunctionSymbolBytes
} derive(Debug, Eq)

///|
/// Syntactic rules for an inert CUDA function-symbol identity.
pub(all) enum FunctionSymbolRule {
  Empty
  TooLong
  NonAscii
  InvalidInitial
  InvalidContinuation
} derive(Debug, Eq)

///|
/// Payload-safe startup failures for content-addressed AOT admission.
///
/// Error payloads contain only canonical digests, fixed limits, and fixed
/// validation categories. They never retain module bytes, symbols, paths,
/// compiler commands, or vendor diagnostics.
pub(all) suberror KernelArtifactError {
  InvalidLimit(KernelArtifactLimit)
  ArtifactCountExceeded(maximum~ : Int)
  EmptyModule(@catalog.AotArtifactDigest)
  ModuleTooLarge(digest~ : @catalog.AotArtifactDigest, maximum~ : Int64)
  TotalModuleBytesExceeded(maximum~ : Int64)
  InvalidFunctionSymbol(
    digest~ : @catalog.AotArtifactDigest,
    rule~ : FunctionSymbolRule
  )
  DigestMismatch(@catalog.AotArtifactDigest)
  DuplicateArtifact(@catalog.AotArtifactDigest)
  MissingArtifact(@catalog.AotArtifactDigest)
  UnreferencedArtifact(@catalog.AotArtifactDigest)
} derive(Debug, Eq)
