///|
/// Stable startup area for local tensor-parallel admission failures.
pub(all) enum TensorParallelAdmissionArea {
  Declaration
  Support
  Topology
  RankBinding
  RankMemory
  Aggregate
} derive(Debug, Eq)

///|
/// Payload-safe rejection reason. No device name, vendor diagnostic, path, or
/// model data enters this vocabulary.
pub(all) enum TensorParallelAdmissionReason {
  InvalidCount
  InvalidRank
  InvalidWorld
  InvalidOrdinal
  InvalidDeviceIdentity
  InvalidPeerCount
  InvalidMemory
  CrossNodeUnsupported
  PipelineUnsupported
  CountMismatch
  RankMismatch
  WorldMismatch
  OrdinalMismatch
  DeviceIdentityMismatch
  TargetMismatch
  IncompletePeerLinks
  CeilingExceedsPhysical
  ReservationExceedsCeiling
  Overflow
} derive(Debug, Eq)

///|
/// Bounded startup rejection. `rank` is an array position or `-1` when the
/// rejection applies to the complete topology.
pub(all) suberror TensorParallelAdmissionError {
  Rejected(
    area~ : TensorParallelAdmissionArea,
    reason~ : TensorParallelAdmissionReason,
    rank~ : Int
  )
} derive(Debug, Eq)

///|
fn reject(
  area : TensorParallelAdmissionArea,
  reason : TensorParallelAdmissionReason,
  rank : Int,
) -> TensorParallelAdmissionError {
  let bounded_rank = if rank >= 0 && rank < MAX_TENSOR_PARALLEL_RANKS {
    rank
  } else {
    -1
  }
  Rejected(area~, reason~, rank=bounded_rank)
}
