///|
/// Bounded policy-file limit categories.
pub(all) enum InstancePolicyLimit {
  PolicyBytes
  JsonDepth
} derive(Debug, Eq)

///|
/// Stable schema sections; no field names or payload text escape.
pub(all) enum InstancePolicyArea {
  Root
  Tokenizer
  Scheduler
  Cache
  Service
  ExternalProtocol
  OpenAiService
  WorkerProcess
  Restart
  Transport
  Preparation
  Telemetry
} derive(Debug, Eq)

///|
/// Payload-safe structural failure categories.
pub(all) enum InstancePolicyIssue {
  ExpectedObject
  ExpectedString
  ExpectedInteger
  ExpectedBoolean
  UnsupportedValue
} derive(Debug, Eq)

///|
/// Payload-safe approved-file operation categories.
pub(all) enum InstancePolicyFileOperation {
  BindRoot
  Open
  Snapshot
  Close
} derive(Debug, Eq)

///|
/// Fail-closed policy snapshot and strict-schema failures.
pub(all) suberror InstancePolicyFileError {
  InvalidExpectedDigest
  InvalidLimit(InstancePolicyLimit)
  FileUnavailable
  UnsupportedFileType
  FileTooLarge(maximum~ : Int64)
  FileTruncated
  FileChangedDuringRead
  FileBusy(InstancePolicyFileOperation)
  FileOperationFailed(InstancePolicyFileOperation)
  DigestMismatch
  InvalidUtf8
  MalformedJson
  JsonDepthExceeded(maximum~ : Int)
  DuplicateField(depth~ : Int, index~ : Int)
  UnknownField(area~ : InstancePolicyArea, index~ : Int)
  MissingField(area~ : InstancePolicyArea, index~ : Int)
  InvalidField(
    area~ : InstancePolicyArea,
    issue~ : InstancePolicyIssue,
    index~ : Int
  )
} derive(Debug, Eq)
