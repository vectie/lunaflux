///|
/// Stable fields used by payload-safe scheduler validation failures.
pub(all) enum SchedulerField {
  Request
  ModelIdentity
  Deadline
  Context
  RequestCapacity
  WaitingCapacity
  ActiveCapacity
  OutputCapacity
  RequestGeneration
  RequestState
  Completion
  PageCapacity
  Plan
} derive(Debug, Eq)

///|
/// Stable validation categories. No variant contains prompt text or tokens.
pub(all) enum SchedulerIssue {
  Duplicate
  Mismatch
  Expired
  Overflow
  Exhausted
  OutOfRange
  Unknown
  Stale
  Backpressured
  InvalidTransition
} derive(Debug, Eq)

///|
/// Bounded scheduler failure without request payloads, paths, or backend text.
pub(all) suberror SchedulerError {
  Invalid(field~ : SchedulerField, issue~ : SchedulerIssue)
  PageAllocator(@page_allocator.PageAllocatorError)
  BlockTable(@block_table.BlockTableError)
  WorkerProtocol(@worker_protocol.WorkerProtocolError)
} derive(Debug, Eq)

///|
fn invalid(field : SchedulerField, issue : SchedulerIssue) -> SchedulerError {
  Invalid(field~, issue~)
}
