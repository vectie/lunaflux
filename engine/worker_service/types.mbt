///|
const FLIGHT_EMPTY : Int = 0

///|
const FLIGHT_PLANNED : Int = 1

///|
const FLIGHT_SUBMITTED : Int = 2

///|
const FLIGHT_RECEIVED : Int = 3

///|
pub(all) enum WorkerServiceLifecycle {
  ServiceReady
  ServiceRecovering
  ServiceClosingFailed
  ServiceClosed
} derive(Debug, Eq)

///|
pub(all) enum WorkerServiceRule {
  Identity
  Generation
  Sequence
  Lifecycle
  Recovery
} derive(Debug, Eq)

///|
pub(all) suberror WorkerServiceError {
  Invalid(WorkerServiceRule)
  Process(@worker_process.WorkerProcessError)
  Scheduler(@core.SchedulerError)
} derive(Debug, Eq)

///|
/// Allocation-free result of one scheduler-to-worker submission attempt.
#valtype
pub enum WorkerServiceDispatch {
  DispatchIdle
  DispatchBackpressured
  DispatchSubmitted(UInt64)
}

///|
pub(all) enum WorkerServiceRestart {
  Restarted
  RestartCleanupRequired(@worker_process.FailedWorkerProcessStartup)
}

///|
/// Thread-confined owner joining one scheduler to one isolated worker process.
pub struct WorkerService {
  priv scheduler : @core.Scheduler
  priv mut process : @worker_process.WorkerProcessSupervisor
  priv flight_states : FixedArray[Int]
  priv flight_sequences : FixedArray[UInt64]
  priv mut lifecycle : WorkerServiceLifecycle
}

///|
pub fn WorkerServiceDispatch::is_idle(self : WorkerServiceDispatch) -> Bool {
  self is DispatchIdle
}

///|
pub fn WorkerServiceDispatch::is_backpressured(
  self : WorkerServiceDispatch,
) -> Bool {
  self is DispatchBackpressured
}

///|
pub fn WorkerServiceDispatch::is_submitted(
  self : WorkerServiceDispatch,
) -> Bool {
  self is DispatchSubmitted(_)
}

///|
pub fn WorkerServiceDispatch::sequence_value(
  self : WorkerServiceDispatch,
) -> UInt64 raise WorkerServiceError {
  match self {
    DispatchSubmitted(sequence) => sequence
    _ => raise Invalid(Sequence)
  }
}
