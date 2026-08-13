///|
pub(all) enum ApprovedFsRule {
  RootPath
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
  Truncated
  Busy(ApprovedFsOperation)
  OperationFailed(ApprovedFsOperation)
} derive(Debug, Eq)
