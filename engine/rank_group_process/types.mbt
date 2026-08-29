///|
pub(all) enum RankGroupProcessRule {
  Limit
  Binding
  Lifecycle
  Owner
  Sequence
  Deadline
  Submission
  Completion
  SchedulerCommit
  Cleanup
} derive(Debug, Eq)

///|
pub(all) enum RankGroupProcessFailure {
  Spawn(rank~ : Int)
  StartupTransport(rank~ : Int)
  TransportLost(rank~ : Int)
  CollectiveFailed(rank~ : Int)
  RankExecutionFailed(rank~ : Int)
  IdleRankExit(rank~ : Int, kind~ : Int, code~ : Int)
  Wire(@rank_group_wire.RankGroupWireError)
  Protocol(@rank_group_protocol.RankGroupError)
  WorkerWire(@worker_wire.WorkerWireError)
  GraphTelemetry(rank~ : Int)
  RankConfigure(
    @tensor_parallel_rank_configure.TensorParallelRankConfigureError
  )
  Clock
  DeadlineExpired
} derive(Debug, Eq)

///|
pub(all) enum RankGroupProcessCleanupFailure {
  RankChildCleanupFailed(rank~ : Int)
} derive(Debug, Eq)

///|
pub(all) suberror RankGroupProcessError {
  Invalid(RankGroupProcessRule)
  Backpressured
  Failed(RankGroupProcessFailure)
} derive(Debug, Eq)

///|
pub(all) enum RankGroupProcessLifecycle {
  Prepared
  Ready
  Submitting
  Polling
  CompletionReady
  CompletionAccepted
  Aborting
  Draining
  Closing
  Faulted
  Cleaning
  Closed
} derive(Debug, Eq)

///|
#valtype
pub(all) enum RankGroupProcessProgress {
  ProcessAdvanced
  ProcessPending
  ProcessCompletionReady
} derive(Debug, Eq)

///|
#valtype
pub(all) enum RankGroupProcessCleanupProgress {
  CleanupAdvanced
  CleanupPending
  CleanupStuck(rank~ : Int)
  CleanupComplete
} derive(Debug, Eq)

///|
pub struct RankGroupProcessLimits {
  priv max_payload_bytes : Int
  priv startup_io_timeout_millis : Int
  priv io_timeout_millis : Int
  priv shutdown_timeout_millis : Int
  priv collective_exchange_timeout_millis : Int
}

///|
/// One exact ordered, canonically typed rank-local Configure contract.
pub struct RankGroupProcessRankStartup {
  priv configure : @tensor_parallel_rank_configure.TensorParallelRankConfigureContract
}

///|
priv enum RankGroupChildOwner {
  PreparedChild(@process.PreparedChildProcess)
  LiveChild(@process.ChildProcess)
}

///|
#valtype
priv enum RankGroupChildCleanupProgress {
  ChildCleanupPending
  ChildCleanupAdvanced
  ChildCleanupStuck
  ChildCleanupComplete
}

///|
pub struct RankGroupProcessSupervisor {
  priv limits : RankGroupProcessLimits
  priv wire_limits : @rank_group_wire.RankGroupWireLimits
  priv children : FixedArray[RankGroupChildOwner]
  priv startups : FixedArray[RankGroupProcessRankStartup]
  priv cleanup_started : FixedArray[Bool]
  priv plan_identities : FixedArray[@worker_protocol.PlanBufferIdentity]
  priv protocol : @rank_group_protocol.RankGroupOwner
  priv wire_transcript : @rank_group_wire.RankGroupWireTranscript
  priv plan_frame : @worker_wire.PlanFrameBuffer
  priv completion_frame : @worker_wire.CompletionFrameBuffer
  priv tx : FixedArray[Byte]
  priv rx : FixedArray[Byte]
  priv rx_staging : FixedArray[Byte]
  priv clock : @monotonic_clock.MonotonicClock
  priv mut activation_consumed : Bool
  priv mut lifecycle_value : RankGroupProcessLifecycle
  priv mut failure_value : RankGroupProcessFailure?
  priv mut cleanup_failure_value : RankGroupProcessCleanupFailure?
  priv mut group_failure : @rank_group_protocol.RankGroupFailureReport?
  priv mut group_failure_taken : Bool
  priv mut io_rank : Int
  priv mut io_phase : Int
  priv mut next_rank : Int
  priv rank_done : FixedArray[Bool]
  priv mut active_slot_code : Int
  priv mut active_sequence_value : UInt64
  priv mut encoded_plan_length : Int
  priv mut active_frame_deadline_millis : Int64
  priv mut leader_received : Bool
  priv graph_telemetry : @worker_wire.WorkerGraphTelemetrySlot
  priv last_graph_telemetry : @worker_wire.WorkerGraphTelemetrySlot
  priv rank_sidecar_pending : FixedArray[Bool]
  priv mut exchange_epoch : UInt64
  priv mut accepted_epoch : UInt64
  priv mut active_deadline_millis : UInt64
  priv mut cleanup_cursor : Int
  priv cleanup_lost : FixedArray[Bool]
  priv mut healthy_terminal : Bool
  priv mut survivor_terminal : Bool
  priv mut survivor_phase : Int
  priv mut survivor_rank : Int
}

///|
pub struct PreparedRankGroupProcess(RankGroupProcessSupervisor)

///|
#valtype
pub(all) enum RankGroupProcessPreparation {
  ReadyRankGroupProcess(RankGroupProcessSupervisor)
  RankGroupProcessCleanupRequired(RankGroupProcessSupervisor)
}

///|
#valtype
pub struct ReceivedRankGroupCompletion(RankGroupProcessSupervisor, UInt64)

///|
#valtype
pub struct AcceptedRankGroupCompletion(RankGroupProcessSupervisor, UInt64)
