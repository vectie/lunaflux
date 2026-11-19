///|
pub(all) enum RuntimeDescriptorLimit {
  DescriptorBytes
  JsonDepth
} derive(Debug, Eq)

///|
pub(all) enum RuntimeDescriptorArea {
  Root
  Model
  Kernels
  Execution
  WorkerLimits
  InferenceLimits
  Ceilings
  TensorParallel
  TensorParallelRank
} derive(Debug, Eq)

///|
pub(all) enum RuntimeDescriptorIssue {
  ExpectedObject
  ExpectedString
  ExpectedInteger
  ExpectedNumber
  ExpectedBoolean
  UnsupportedValue
} derive(Debug, Eq)

///|
pub(all) enum RuntimeDescriptorFileOperation {
  Open
  Snapshot
  Close
  BindModelRoot
  BindKernelRoot
} derive(Debug, Eq)

///|
pub(all) enum RuntimeDescriptorAdmissionStage {
  KernelConfig
  ModelIdentity
  ModelConfig
  ModelPlan
  WorkerLimits
  InferenceLimits
  DeviceTarget
  KvLayout
  WeightInspection
  DeviceStepLimits
  ExecutionLimits
  ExecutionManifest
  BootstrapManifest
  BootstrapDigest
  BootstrapSource
  StartupContract
  DeviceWorkerPlan
  NumericWeightInspection
  DeviceCapability
  DeviceNumericCapability
  I8ProductionPolicy
  WeightArtifactDigest
  GraphMemory
  TensorParallelTopology
  TensorParallelSource
  TensorParallelInspection
} derive(Debug, Eq)

///|
/// Payload-safe failure vocabulary for runtime-descriptor admission.
/// No variant contains a path, field name, file bytes, native error, or device
/// handle.
pub(all) suberror RuntimeDescriptorFileError {
  InvalidExpectedDigest
  InvalidLimit(RuntimeDescriptorLimit)
  FileUnavailable
  UnsupportedFileType
  FileTooLarge(maximum~ : Int64)
  FileTruncated
  FileChangedDuringRead
  FileBusy(RuntimeDescriptorFileOperation)
  FileOperationFailed(RuntimeDescriptorFileOperation)
  DigestMismatch
  InvalidUtf8
  MalformedJson
  JsonDepthExceeded(maximum~ : Int32)
  DuplicateField(depth~ : Int32, index~ : Int32)
  UnknownField(area~ : RuntimeDescriptorArea, index~ : Int32)
  MissingField(area~ : RuntimeDescriptorArea, index~ : Int32)
  InvalidField(
    area~ : RuntimeDescriptorArea,
    issue~ : RuntimeDescriptorIssue,
    index~ : Int32
  )
  AdmissionRejected(RuntimeDescriptorAdmissionStage)
} derive(Debug, Eq)
