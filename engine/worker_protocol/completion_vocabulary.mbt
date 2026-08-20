///|
/// Backend-neutral bounded failure category for one row. Raw device or vendor
/// diagnostics stay behind the worker boundary.
pub(all) enum WorkerFailure {
  ExecutionFailed
  ResourceUnavailable
  CancelledBeforeExecution
} derive(Debug, Eq)
