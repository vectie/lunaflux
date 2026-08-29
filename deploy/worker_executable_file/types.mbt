///|
/// Exact independently supplied worker executable identity.
pub struct WorkerExecutableDigest(String) derive(Debug, Eq)

///|
/// Startup bound for hashing one executable snapshot.
pub struct WorkerExecutableFileLimits {
  priv maximum_bytes : Int64
}

///|
/// Opaque identity plus the exact pinned executable object used by process
/// activation. The path label is retained only for deployment evidence; it is
/// never reopened to authorize a child.
pub struct WorkerExecutableAdmission {
  priv activation_path : Bytes
  priv digest : WorkerExecutableDigest
  priv handle : @cap.ApprovedExecutableHandle
}

///|
/// Root-free evidence for an offline materialized executable. It deliberately
/// cannot be passed to any live activation API.
pub struct MaterializedWorkerExecutableEvidence {
  priv activation_path : Bytes
  priv digest : WorkerExecutableDigest
}

///|
/// Payload-safe executable admission failures.
pub(all) suberror WorkerExecutableFileError {
  InvalidPath
  InvalidDigest
  InvalidLimit
  FileUnavailable
  UnsupportedFileType
  FileTooLarge
  FileTruncated
  FileChangedDuringRead
  FileBusy
  FileOperationFailed
  DigestMismatch
  UnsupportedPlatform
} derive(Debug, Eq)
