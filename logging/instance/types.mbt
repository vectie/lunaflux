///|
const LUNA_INSTANCE_LOG_MAX_CAPACITY : Int = 65536

///|
const LUNA_INSTANCE_LOG_MAX_COUNT : Int = 1000000

///|
const LUNA_INSTANCE_LOG_MAX_DURATION_MILLIS : Int = 86400000

///|
#valtype
pub(all) enum LunaInstanceLogEvent {
  LunaInstanceStarting
  LunaInstanceReady
  LunaAdmissionAccepted
  LunaAdmissionRejected
  LunaRequestCompleted
  LunaRequestCancelled
  LunaRequestDeadline
  LunaRequestFailed
  LunaWorkerRestarted
  LunaWorkerFailed
  LunaNetworkAccepted
  LunaNetworkDisconnected
  LunaNetworkRejected
  LunaBackpressureObserved
  LunaDrainStarted
  LunaInstanceClosed
} derive(Eq)

///|
#valtype
pub(all) enum LunaInstanceLogReason {
  LunaLogNoReason
  LunaLogAuthentication
  LunaLogValidation
  LunaLogCapacity
  LunaLogDeadline
  LunaLogCancellation
  LunaLogWorkerExit
  LunaLogWorkerProtocol
  LunaLogNetworkRead
  LunaLogNetworkWrite
  LunaLogPeerClosed
  LunaLogServiceDraining
  LunaLogInternal
} derive(Eq)

///|
pub(all) enum LunaInstanceLogRule {
  LunaInstanceLogCapacityRule
  LunaInstanceLogIndexRule
  LunaInstanceLogTimestampRule
  LunaInstanceLogCountRule
  LunaInstanceLogDurationRule
  LunaInstanceLogSnapshotRule
} derive(Eq)

///|
pub(all) suberror LunaInstanceLogError {
  LunaInstanceLogInvalid(LunaInstanceLogRule)
  LunaInstanceLogStaleSnapshot
  LunaInstanceLogSnapshotGenerationExhausted
} derive(Eq)

///|
/// Startup-owned fixed-capacity structured event ring for one instance.
pub struct LunaInstanceLog {
  priv events : FixedArray[Int]
  priv reasons : FixedArray[Int]
  priv timestamps : FixedArray[UInt64]
  priv counts : FixedArray[Int]
  priv durations : FixedArray[Int]
  priv snapshot_events : FixedArray[Int]
  priv snapshot_reasons : FixedArray[Int]
  priv snapshot_timestamps : FixedArray[UInt64]
  priv snapshot_counts : FixedArray[Int]
  priv snapshot_durations : FixedArray[Int]
  priv mut head : Int
  priv mut length : Int
  priv mut dropped_count : UInt64
  priv mut has_timestamp : Bool
  priv mut last_timestamp : UInt64
  priv mut snapshot_length : Int
  priv mut snapshot_dropped_count : UInt64
  priv mut snapshot_generation : UInt64
}

///|
/// Opaque generation-authenticated view of the one preallocated snapshot.
#valtype
pub struct LunaInstanceLogSnapshot {
  priv owner : LunaInstanceLog
  priv generation : UInt64
}
