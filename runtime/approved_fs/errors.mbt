///|
pub(all) enum ApprovedFsRule {
  RootPath
  RootIdentity
  RelativeLocator
  ReadRange
  Lifecycle
} derive(Debug, Eq)

///|
pub(all) enum ApprovedFsObject {
  Root
  File
} derive(Debug, Eq)

///|
pub(all) enum ApprovedFsOperation {
  Open
  Read
  Stat
  Close
} derive(Debug, Eq)

///|
/// Stable payload-safe failures. No case retains a path, descriptor, native
/// error number, file content, or platform message.
pub(all) suberror ApprovedFsError {
  Invalid(ApprovedFsRule)
  Unavailable(ApprovedFsObject)
  UnsupportedType(ApprovedFsObject)
  FileTooLarge
  ChangedDuringRead
  Truncated
  Busy(ApprovedFsOperation)
  OperationFailed(ApprovedFsOperation)
} derive(Debug, Eq)

///|
pub extend ApprovedFsRule with @debug.Debug::{to_repr}

///|
pub extend ApprovedFsRule with Eq::{not_equal, equal}

///|
pub extend ApprovedFsObject with @debug.Debug::{to_repr}

///|
pub extend ApprovedFsObject with Eq::{not_equal, equal}

///|
pub extend ApprovedFsOperation with @debug.Debug::{to_repr}

///|
pub extend ApprovedFsOperation with Eq::{not_equal, equal}

///|
pub extend ApprovedFsError with @debug.Debug::{to_repr}

///|
pub extend ApprovedFsError with Eq::{not_equal, equal}
