///|
const RANK_GROUP_WIRE_MAX_WORLD_SIZE : Int = 16

///|
const RANK_GROUP_WIRE_MAX_PAYLOAD_BYTES : Int = 67108220

///|
const RANK_GROUP_WIRE_HEADER_BYTES : Int = 644

///|
pub(all) enum RankGroupWireDirection {
  ParentToChild
  ChildToParent
} derive(Debug, Eq)

///|
pub(all) enum RankGroupWireSlot {
  A
  B
} derive(Debug, Eq)

///|
pub(all) enum RankGroupWirePlanFailure {
  CollectiveFailure
  RankExecutionFailure
} derive(Debug, Eq)

///|
pub(all) enum RankGroupWireKind {
  Configure
  Ready
  Submit
  PollComplete
  Pending
  Complete
  PlanFailed(RankGroupWirePlanFailure)
  Abort
  Aborted
  Drain
  Drained
  Close
  Closed
} derive(Debug, Eq)

///|
pub(all) enum RankGroupWireLifecycle {
  NewGroup
  Configuring
  ReadyGroup
  Executing
  Polling
  Aborting
  AbortedPlan
  Draining
  DrainedGroup
  Closing
  ClosedGroup
  Faulted
} derive(Debug, Eq)

///|
pub(all) enum RankGroupWireRule {
  Limit
  Magic
  Version
  Length
  Checksum
  Reserved
  Kind
  Direction
  Slot
  Payload
  Digest
  ModelIdentity
  ModelGeneration
  GroupGeneration
  Rank
  WorldSize
  Sequence
  Binding
  Lifecycle
  TransportLost
  FrameEpoch
  Destination
} derive(Debug, Eq)

///|
pub(all) suberror RankGroupWireError {
  Invalid(RankGroupWireRule)
  ProtocolFault(RankGroupWireRule)
  TranscriptFaulted(RankGroupWireRule)
} derive(Debug, Eq)

///|
pub struct RankGroupWireLimits {
  priv max_payload_bytes : Int
}

///|
pub struct RankGroupWireBinding {
  priv identity : @spec.ModelIdentity
  priv model_generation_value : UInt64
  priv group_generation_value : UInt64
  priv rank : Int
  priv world_size : Int
  priv group_digest : String
  priv topology_digest : String
  priv worker_digest : String
  priv device_plan_digest : String
  priv execution_plan_digest : String
  priv collective_digest : String
  priv kv_digest : String
  priv canonical_digests : FixedArray[Byte]
}

///|
pub struct RankGroupWireFrameBuffer {
  priv limits : RankGroupWireLimits
  priv storage : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut length : Int
}

///|
#valtype
pub struct ValidatedRankGroupWireFrame(RankGroupWireFrameBuffer, UInt64)

///|
pub struct RankGroupWireTranscript {
  priv bindings : FixedArray[RankGroupWireBinding]
  priv rank_states : FixedArray[Int]
  priv configured_ranks : FixedArray[Bool]
  priv world_size : Int
  priv mut lifecycle : RankGroupWireLifecycle
  priv mut previous_sequence_value : UInt64
  priv mut active_sequence_value : UInt64
  priv mut active_slot_code : Int
  priv mut phase_count : Int
  priv mut phase_target_count : Int
  priv mut fault_rule : RankGroupWireRule
  priv mut plan_failure_value : RankGroupWirePlanFailure?
  priv mut plan_failure_rank_value : Int
}
