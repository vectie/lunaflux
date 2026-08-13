///|
/// A path-bearing file is represented only by its manifest position.
pub(all) enum ArtifactFileRef {
  ManifestFile
  ModuleFile(Int)
} derive(Debug, Eq)

///|
/// Lexical path-policy failures. The rejected path is never retained.
pub(all) enum ArtifactPathIssue {
  EmptyPath
  PathTooLong
  AbsolutePath
  TraversalSegment
  InvalidPathCharacter
  DuplicatePath
} derive(Debug, Eq)

///|
pub(all) enum ManifestArea {
  ManifestRoot
  ManifestModelIdentity
  ManifestDeviceTarget
  ManifestModule
  ManifestEntryPoint
} derive(Debug, Eq)

///|
pub(all) enum ManifestField {
  SchemaVersionField
  ModelIdentityField
  ContentSha256Field
  PlanSha256Field
  DeviceTargetField
  ComputeMajorField
  ComputeMinorField
  SupportsBf16Field
  SupportsCublasLtField
  CatalogVersionField
  ModulesField
  ModuleSha256Field
  ModulePathField
  EntryPointsField
  FamilyIdField
  EntryPointIdField
  FunctionSymbolField
} derive(Debug, Eq)

///|
pub(all) enum ManifestValueIssue {
  ExpectedObject
  ExpectedArray
  ExpectedString
  ExpectedInteger
  ExpectedBoolean
  UnsupportedValue
  InvalidSha256
  CountExceeded
  StringTooLong
} derive(Debug, Eq)

///|
pub(all) enum ManifestIdentityPart {
  ModelContentIdentity
  ModelPlanIdentity
  DeviceComputeMajor
  DeviceComputeMinor
  DeviceBf16Support
  DeviceCublasLtSupport
  KernelCatalogVersion
} derive(Debug, Eq)

///|
pub(all) enum ArtifactFileLimit {
  ManifestBytes
  JsonDepth
  PathBytes
} derive(Debug, Eq)

///|
/// Bounded collection within one inert module-file source declaration.
pub(all) enum ArtifactFileSourceCount {
  SourceModules
  SourceEntryPoints
} derive(Debug, Eq)

///|
/// Bounded, payload-safe failures from AOT manifest and module-file admission.
///
/// No variant retains a filesystem path, file contents, function symbol,
/// environment value, or raw I/O error.
pub(all) suberror KernelArtifactFileError {
  InvalidExpectedManifestDigest
  InvalidLimit(ArtifactFileLimit)
  SourceCountExceeded(source~ : ArtifactFileSourceCount, maximum~ : Int)
  InvalidPath(file~ : ArtifactFileRef, issue~ : ArtifactPathIssue)
  ApprovedRootUnavailable
  UnsupportedApprovedRootType
  FileUnavailable(ArtifactFileRef)
  UnsupportedPathType(ArtifactFileRef)
  FileTooLarge(file~ : ArtifactFileRef, maximum~ : Int64)
  TotalModuleBytesExceeded(maximum~ : Int64)
  FileTruncated(ArtifactFileRef)
  FileChangedDuringRead(ArtifactFileRef)
  DigestMismatch(ArtifactFileRef)
  InvalidUtf8
  MalformedJson
  JsonDepthExceeded(maximum~ : Int)
  DuplicateField(depth~ : Int, index~ : Int)
  UnknownField(area~ : ManifestArea, index~ : Int)
  MissingField(area~ : ManifestArea, field~ : ManifestField)
  InvalidField(
    area~ : ManifestArea,
    field~ : ManifestField,
    issue~ : ManifestValueIssue,
    index~ : Int
  )
  IdentityMismatch(ManifestIdentityPart)
  ArtifactRejected(@artifact.KernelArtifactError)
} derive(Debug, Eq)
