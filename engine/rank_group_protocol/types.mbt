///|
const MAX_RANK_GROUP_WORLD_SIZE : Int = 16

///|
pub(all) enum RankGroupLifecycle {
  Ready
  Executing
  Completed
  Faulted
  Closed
} derive(Debug, Eq)

///|
pub(all) enum RankGroupRule {
  WorldSize
  Rank
  LeaderRank
  FollowerRank
  GroupGeneration
  GroupEpoch
  ModelPlanGeneration
  PlanSequence
  Submission
  PlanBufferSet
  Completion
  DuplicateFollowerAck
  FollowersPending
  Deadline
  TimeRegression
  FailurePublication
  LostRankDrain
  DuplicateDrainAck
  DrainPending
} derive(Debug, Eq)

///|
pub(all) enum RankGroupFailureReason {
  ProtocolViolation(RankGroupRule)
  DeadlineExpired
  RankLost
  CollectiveFailure
  RankExecutionFailure
  HostFailure
} derive(Debug, Eq)

///|
pub(all) suberror RankGroupError {
  Invalid(RankGroupRule)
  InvalidLifecycle(expected~ : RankGroupLifecycle, actual~ : RankGroupLifecycle)
  ProtocolFault(RankGroupRule)
  GroupFaulted(RankGroupFailureReason)
  NotReady(RankGroupRule)
} derive(Debug, Eq)

///|
pub struct RankGroupGeneration {
  priv value : UInt64
} derive(Debug, Eq, Hash)

///|
pub struct RankGroupOwner {
  priv identity : @spec.ModelIdentity
  priv model_plan_generation_value : UInt64
  priv generation : RankGroupGeneration
  priv world_size : Int
  priv plan_buffer_identities : FixedArray[@worker_protocol.PlanBufferIdentity]
  priv mut lifecycle : RankGroupLifecycle
  priv mut epoch : UInt64
  priv mut previous_sequence_value : UInt64
  priv mut active_sequence_value : UInt64
  priv mut active_started_tick : UInt64
  priv mut active_deadline_tick : UInt64
  priv mut active_plan_buffer_index : Int
  priv follower_acks : FixedArray[Bool]
  priv drained_ranks : FixedArray[Bool]
  priv lost_ranks : FixedArray[Bool]
  priv mut follower_ack_count : Int
  priv mut failure_reason : RankGroupFailureReason
  priv mut failure_rank : Int
  priv mut failure_published : Bool
  priv mut drain_count : Int
}

///|
/// Exact rank-group publication for one authenticated semantic worker plan.
/// Its private constructor prevents a caller from substituting a different
/// plan owner that merely repeats the same scalar sequence.
#valtype
pub struct SubmittedRankGroupPlan(RankGroupOwner, UInt64)

///|
/// Immutable rank-specific view of one exact group-plan epoch.
#valtype
pub struct RankPlanLease(RankGroupOwner, UInt64, Int)

///|
/// Payload-free follower result suitable for fixed scalar framing.
#valtype
pub struct RankExecutionAck(UInt64, UInt64, UInt64, Int, Int)

///|
/// Authenticated successful group result. Only rank zero contributes the
/// canonical completion payload; every follower has already acknowledged.
#valtype
pub struct CompletedRankGroupPlan(RankGroupOwner, UInt64)

///|
/// One-shot payload-safe terminal failure publication.
#valtype
pub struct RankGroupFailureReport(
  UInt64,
  UInt64,
  RankGroupFailureReason,
  Int,
  Int
)

///|
/// Rank-specific authority to drain one survivor after group invalidation.
#valtype
pub struct RankDrainLease(RankGroupOwner, UInt64, Int)

///|
/// Payload-free survivor drain result suitable for fixed scalar framing.
#valtype
pub struct RankDrainAck(UInt64, UInt64, Int, Int)
