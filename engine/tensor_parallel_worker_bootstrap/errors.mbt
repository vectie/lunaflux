///|
/// Payload-free stages for tensor-parallel group-bootstrap rejection.
pub(all) enum TensorParallelWorkerBootstrapRule {
  FrameMagic
  FrameVersion
  FrameLength
  FrameChecksum
  FrameReserved
  Digest
  ModelIdentity
  ModelGeneration
  GroupGeneration
  WorldSize
  Rank
  DeviceOrdinal
  Topology
  WorkerContract
  DevicePlan
  ExecutionPlan
  CollectiveContract
  KvContract
  RendezvousId
  DuplicateRank
  IncompleteRankSet
} derive(Debug, Eq)

///|
/// Bounded startup failures. No model path, device name, request payload, or
/// backend error text crosses this package boundary.
pub(all) suberror TensorParallelWorkerBootstrapError {
  Rejected(rule~ : TensorParallelWorkerBootstrapRule, rank~ : Int)
  DestinationTooSmall
  FrameEpochOverflow
  StaleFrame
} derive(Debug, Eq)

///|
fn rejected(
  rule : TensorParallelWorkerBootstrapRule,
  rank : Int,
) -> TensorParallelWorkerBootstrapError {
  Rejected(rule~, rank~)
}
