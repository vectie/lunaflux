///|
pub(all) enum ModelConfigFileLimit {
  ConfigBytes
  JsonDepth
} derive(Debug, Eq)

///|
pub(all) enum ModelConfigFileOperation {
  OpenFile
  SnapshotFile
  CloseFile
} derive(Debug, Eq)

///|
pub(all) enum ModelConfigFileIoIssue {
  AuthorityClosed
  Busy
  Failed
} derive(Debug, Eq)

///|
/// Payload-safe failures from bounded model-configuration admission.
pub(all) suberror ModelConfigFileError {
  InvalidExpectedDigest
  InvalidLimit(ModelConfigFileLimit)
  FileUnavailable
  UnsupportedPathType
  FileIo(operation~ : ModelConfigFileOperation, issue~ : ModelConfigFileIoIssue)
  FileTooLarge(maximum~ : Int64)
  FileTruncated
  FileChangedDuringRead
  DigestMismatch
  InvalidConfiguration(@config_reader.ModelConfigReaderError)
} derive(Debug, Eq)
