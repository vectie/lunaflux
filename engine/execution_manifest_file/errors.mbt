///|
pub(all) enum ExecutionManifestLimit {
  ManifestBytes
  JsonDepth
  ModuleCount
  OperationCount
  DeviceMemory
  LaunchContract
  ArtifactFile
} derive(Debug, Eq)

///|
pub(all) enum ExecutionManifestArea {
  Root
  ModelIdentity
  DeviceTarget
  Profile
  Module
  Operation
  Dimensions
  LunaAuthorization
  NumericAuthorization
} derive(Debug, Eq)

///|
pub(all) enum ExecutionManifestIssue {
  ExpectedObject
  ExpectedArray
  ExpectedString
  ExpectedInteger
  ExpectedBoolean
  InvalidSha256
  UnsupportedValue
  CountExceeded
  InvalidOrder
  DuplicateValue
} derive(Debug, Eq)

///|
pub(all) enum ExecutionManifestIdentityPart {
  ContentDigest
  PlanDigest
  ComputeMajor
  ComputeMinor
  Bf16Support
  CublasLtSupport
  CatalogVersion
  NumericSchemaDigest
  I8AdmissionDigest
  DeviceNumericDigest
  WeightArtifactDigest
  Fp8KernelManifestDigest
  Fp8RuntimeRecipeDigest
  Fp8ReusableRecipeDigest
} derive(Debug, Eq)

///|
pub(all) enum ExecutionManifestOperation {
  Open
  Snapshot
  Close
} derive(Debug, Eq)

///|
pub(all) enum ExecutionManifestAdmissionStage {
  Catalog
  StaticPlan
  MemoryPlan
  LaunchContracts
  ArtifactSource
  ArtifactFiles
  TensorParallelSpecialization
  TensorParallelLaunchContracts
  TensorParallelExecutionPlan
  LunaSpecialization
  LunaExecutionProfile
  LunaCapability
  LunaExternalApproval
  LunaArtifactAuthorization
  Blueprint
  I8ProductionPolicy
  I8BootstrapV3
  Fp8NumericLayout
  Fp8KernelManifest
  Fp8StartupV2
  Fp8RuntimeRecipeV2
  Fp8ReusableRecipeV3
} derive(Debug, Eq)

///|
/// Payload-safe failure vocabulary for one execution-manifest admission.
///
/// No variant retains a path, JSON value, symbol, file contents, descriptor,
/// errno, or lower-layer error payload.
pub(all) suberror ExecutionManifestFileError {
  InvalidExpectedDigest
  InvalidLimit(ExecutionManifestLimit)
  FileUnavailable
  UnsupportedFileType
  FileTooLarge(maximum~ : Int64)
  FileTruncated
  FileChangedDuringRead
  FileBusy(ExecutionManifestOperation)
  FileOperationFailed(ExecutionManifestOperation)
  DigestMismatch
  InvalidUtf8
  MalformedJson
  JsonDepthExceeded(maximum~ : Int)
  DuplicateField(depth~ : Int, index~ : Int)
  UnknownField(area~ : ExecutionManifestArea, index~ : Int)
  MissingField(area~ : ExecutionManifestArea, index~ : Int)
  InvalidField(
    area~ : ExecutionManifestArea,
    issue~ : ExecutionManifestIssue,
    index~ : Int
  )
  IdentityMismatch(ExecutionManifestIdentityPart)
  UnsupportedModelEnvelope
  AdmissionRejected(ExecutionManifestAdmissionStage)
} derive(Debug, Eq)
